from flask import Flask, jsonify, send_file, request, redirect
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import boto3
from botocore.exceptions import ClientError
import requests
import os
import tempfile
import config
import cv2
from werkzeug.utils import secure_filename
import subprocess
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Configuración de upload
UPLOAD_FOLDER = '/tmp/penal_uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
# CORS(app)
# Configurar CORS correctamente
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Agregar manejo explícito de OPTIONS para todos los endpoints
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        headers['Access-Control-Allow-Credentials'] = 'true'
        return response

# Cliente S3
if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        's3',
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
    )
else:
    s3_client = boto3.client('s3', region_name=config.AWS_REGION)

# Headers para API-Football
API_FOOTBALL_HEADERS = {
    'x-apisports-key': config.API_FOOTBALL_KEY
}

def get_db_connection():
    """Crea y retorna una conexión a la base de datos PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=config.DB_HOST,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            port=config.DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Error conectando a la base de datos: {e}")
        raise

# ==================== ENDPOINTS DE JUGADORES ====================

@app.route('/api/players', methods=['GET'])
def get_players():
    """Endpoint para obtener todos los jugadores"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                player_id,
                short_name,
                name,
                lastname,
                foot
            FROM players
            ORDER BY lastname, name
        """
        
        cursor.execute(query)
        players = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify(players), 200
        
    except Exception as e:
        print(f"Error en get_players: {e}")
        return jsonify({
            'error': 'Error al obtener jugadores',
            'message': str(e)
        }), 500
    
@app.route('/api/players/stats', methods=['GET'])
def get_players_stats():
    """Endpoint para obtener jugadores con sus estadísticas de penales"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                p.player_id,
                p.short_name,
                p.name,
                p.lastname,
                p.foot,
                COUNT(pen.penalty_id) as total_penalties,
                COUNT(CASE WHEN pen.event = 'Penalty' THEN 1 END) as goals,
                COUNT(CASE WHEN pen.event = 'Missed Penalty' THEN 1 END) as missed,
                CASE 
                    WHEN COUNT(pen.penalty_id) > 0 THEN 
                        ROUND((COUNT(CASE WHEN pen.event = 'Penalty' THEN 1 END)::numeric / COUNT(pen.penalty_id)::numeric) * 100, 1)
                    ELSE 0 
                END as effectiveness
            FROM players p
            LEFT JOIN penalties pen ON p.player_id = pen.player_id
            GROUP BY p.player_id, p.short_name, p.name, p.lastname, p.foot
            ORDER BY p.lastname, p.name
        """
        
        cursor.execute(query)
        players = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify(players), 200
        
    except Exception as e:
        print(f"Error en get_players_stats: {e}")
        return jsonify({
            'error': 'Error al obtener estadísticas de jugadores',
            'message': str(e)
        }), 500

@app.route('/api/players/<int:player_id>', methods=['GET'])
def get_player(player_id):
    """Endpoint para obtener un jugador específico por ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                player_id,
                short_name,
                name,
                lastname,
                foot
            FROM players
            WHERE player_id = %s
        """
        
        cursor.execute(query, (player_id,))
        player = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if player:
            return jsonify(player), 200
        else:
            return jsonify({'error': 'Jugador no encontrado'}), 404
            
    except Exception as e:
        print(f"Error en get_player: {e}")
        return jsonify({
            'error': 'Error al obtener jugador',
            'message': str(e)
        }), 500

# ==================== ENDPOINTS DE PENALTIES ====================

@app.route('/api/penalties/filters', methods=['GET'])
def get_penalty_filters():
    """Obtiene las opciones disponibles para filtros (ligas, temporadas, equipos)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Obtener ligas únicas
        cursor.execute("""
            SELECT DISTINCT l.league_id, l.name, l.season
            FROM leagues l
            INNER JOIN penalties p ON l.league_id = p.league_id AND l.season = p.season
            ORDER BY l.name, l.season DESC
        """)
        leagues = cursor.fetchall()
        
        # Obtener temporadas únicas
        cursor.execute("""
            SELECT DISTINCT season
            FROM penalties
            ORDER BY season DESC
        """)
        seasons = [row['season'] for row in cursor.fetchall()]
        
        # Obtener equipos únicos
        cursor.execute("""
            SELECT DISTINCT t.team_id, t.name
            FROM teams t
            INNER JOIN penalties p ON (t.team_id = p.shooter_team_id OR t.team_id = p.defender_team_id)
            ORDER BY t.name
        """)
        teams = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'leagues': leagues,
            'seasons': seasons,
            'teams': teams
        }), 200
        
    except Exception as e:
        print(f"Error en get_penalty_filters: {e}")
        return jsonify({
            'error': 'Error al obtener filtros',
            'message': str(e)
        }), 500

@app.route('/api/penalties', methods=['GET'])
def get_penalties():
    """Obtiene lista de penales con filtros opcionales"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Obtener parámetros de filtro
        league_id = request.args.get('league_id', type=int)
        season = request.args.get('season', type=int)
        shooter_team_id = request.args.get('shooter_team_id', type=int)
        defender_team_id = request.args.get('defender_team_id', type=int)
        
        # Query base
        query = """
            SELECT 
                p.penalty_id,
                p.fixture_id,
                p.minute,
                p.extra_minute,
                p.condition,
                p.penalty_shootout,
                p.height,
                p.side,
                l.name as league_name,
                l.season,
                st.name as shooter_team_name,
                dt.name as defender_team_name,
                pl.short_name as player_short_name,
                pl.name as player_name,
                pl.lastname as player_lastname
            FROM penalties p
            LEFT JOIN leagues l ON p.league_id = l.league_id AND p.season = l.season
            LEFT JOIN teams st ON p.shooter_team_id = st.team_id
            LEFT JOIN teams dt ON p.defender_team_id = dt.team_id
            LEFT JOIN players pl ON p.player_id = pl.player_id
            WHERE 1=1
        """
        
        params = []
        
        # Aplicar filtros
        if league_id:
            query += " AND p.league_id = %s"
            params.append(league_id)
        
        if season:
            query += " AND p.season = %s"
            params.append(season)
        
        if shooter_team_id:
            query += " AND p.shooter_team_id = %s"
            params.append(shooter_team_id)
        
        if defender_team_id:
            query += " AND p.defender_team_id = %s"
            params.append(defender_team_id)
        
        query += " ORDER BY p.penalty_id DESC LIMIT 100"
        
        cursor.execute(query, params)
        penalties = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify(penalties), 200
        
    except Exception as e:
        print(f"Error en get_penalties: {e}")
        return jsonify({
            'error': 'Error al obtener penales',
            'message': str(e)
        }), 500

@app.route('/api/penalties/<int:penalty_id>', methods=['GET'])
def get_penalty_detail(penalty_id):
    """Obtiene información detallada de un penal específico"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                p.penalty_id,
                p.fixture_id,
                p.minute,
                p.extra_minute,
                p.condition,
                p.penalty_shootout,
                p.height,
                p.side,
                p.event,
                l.name as league_name,
                l.season,
                st.team_id as shooter_team_id,
                st.name as shooter_team_name,
                dt.team_id as defender_team_id,
                dt.name as defender_team_name,
                pl.player_id,
                pl.short_name as player_short_name,
                pl.name as player_name,
                pl.lastname as player_lastname,
                pl.foot as player_foot
            FROM penalties p
            LEFT JOIN leagues l ON p.league_id = l.league_id AND p.season = l.season
            LEFT JOIN teams st ON p.shooter_team_id = st.team_id
            LEFT JOIN teams dt ON p.defender_team_id = dt.team_id
            LEFT JOIN players pl ON p.player_id = pl.player_id
            WHERE p.penalty_id = %s
        """
        
        cursor.execute(query, (penalty_id,))
        penalty = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if penalty:
            return jsonify(penalty), 200
        else:
            return jsonify({'error': 'Penal no encontrado'}), 404
            
    except Exception as e:
        print(f"Error en get_penalty_detail: {e}")
        return jsonify({
            'error': 'Error al obtener detalle del penal',
            'message': str(e)
        }), 500

@app.route('/api/penalties/<int:penalty_id>/postures', methods=['GET'])
def get_penalty_postures(penalty_id):
    """Obtiene todas las posturas (frames) de un penal específico"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT *
            FROM postures
            WHERE penalty_id = %s
            ORDER BY frame
        """
        
        cursor.execute(query, (penalty_id,))
        postures = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if postures:
            return jsonify(postures), 200
        else:
            return jsonify({'error': 'No se encontraron posturas para este penal'}), 404
            
    except Exception as e:
        print(f"Error en get_penalty_postures: {e}")
        return jsonify({
            'error': 'Error al obtener posturas',
            'message': str(e)
        }), 500

@app.route('/api/penalties/<int:penalty_id>/video', methods=['GET'])
def get_penalty_video(penalty_id):
    """Genera una URL pre-firmada de S3 para el video"""
    try:
        video_key = f"{penalty_id}.mp4"
        
        print(f"🔍 Generando URL pre-firmada para: {video_key}")
        
        # Generar URL pre-firmada (válida por 1 hora)
        try:
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': config.S3_BUCKET_NAME,
                    'Key': video_key,
                    'ResponseContentType': 'video/mp4'
                },
                ExpiresIn=3600  # 1 hora
            )
            
            print(f"✅ URL generada exitosamente")
            
            # Devolver la URL como JSON
            return jsonify({
                'video_url': presigned_url,
                'expires_in': 3600
            }), 200
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ Error S3: {error_code}")
            
            if error_code == '404' or error_code == 'NoSuchKey':
                return jsonify({'error': f'Video {video_key} no encontrado en S3'}), 404
            elif error_code == '403' or error_code == 'Forbidden':
                return jsonify({
                    'error': 'Acceso denegado al bucket S3',
                    'message': 'Verifica las credenciales AWS y permisos IAM'
                }), 403
            else:
                raise
        
    except Exception as e:
        print(f"❌ Error en get_penalty_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Error al obtener video',
            'message': str(e)
        }), 500

@app.route('/api/penalties/next-id', methods=['GET'])
def get_next_penalty_id():
    """Obtiene el siguiente ID disponible para penalties"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COALESCE(MAX(penalty_id), 0) + 1 as next_id FROM penalties")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({'next_penalty_id': result[0]}), 200
        
    except Exception as e:
        print(f"Error en get_next_penalty_id: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== ENDPOINTS DE API-FOOTBALL ====================

@app.route('/api/api-football/players/search', methods=['GET'])
def search_players():
    """Busca jugadores primero en BD local, luego en API-Football"""
    try:
        search_query = request.args.get('search', '')
        
        if not search_query or len(search_query) < 2:
            return jsonify({'error': 'La búsqueda debe tener al menos 2 caracteres'}), 400
        
        players = []
        
        # # 1. BUSCAR PRIMERO EN NUESTRA BASE DE DATOS
        # try:
        #     conn = get_db_connection()
        #     cursor = conn.cursor(cursor_factory=RealDictCursor)
            
        #     search_pattern = f"%{search_query}%"
        #     query = """
        #         SELECT 
        #             player_id,
        #             short_name,
        #             name,
        #             lastname,
        #             foot,
        #             'local' as source
        #         FROM players
        #         WHERE LOWER(name) LIKE LOWER(%s)
        #            OR LOWER(lastname) LIKE LOWER(%s)
        #            OR LOWER(short_name) LIKE LOWER(%s)
        #         ORDER BY lastname, name
        #         LIMIT 10
        #     """
            
        #     cursor.execute(query, (search_pattern, search_pattern, search_pattern))
        #     local_players = cursor.fetchall()
            
        #     cursor.close()
        #     conn.close()
            
        #     # Formatear resultados locales
        #     for player in local_players:
        #         players.append({
        #             'player_id': player['player_id'],
        #             'name': player['name'] or '',
        #             'lastname': player['lastname'] or '',
        #             'short_name': player['short_name'] or '',
        #             'nationality': '',
        #             'birth_date': '',
        #             'photo': '',
        #             'source': 'local'
        #         })
            
        #     print(f"🔍 Encontrados {len(local_players)} jugadores en BD local")
            
        # except Exception as e:
        #     print(f"⚠️ Error buscando en BD local: {e}")
        
        # 2. SI NO HAY RESULTADOS LOCALES, BUSCAR EN API-FOOTBALL
        if len(players) == 0:
            print(f"🌐 No hay resultados locales. Buscando en API-Football: {search_query}")
            
            try:
                api_url = f"{config.API_FOOTBALL_URL}/players/profiles"
                params = {'search': search_query}
                
                response = requests.get(
                    api_url, 
                    headers=API_FOOTBALL_HEADERS, 
                    params=params, 
                    timeout=10
                )
                
                print(f"📡 API-Football status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"❌ Error API-Football: {response.status_code}")
                    print(f"Response: {response.text}")
                    return jsonify({
                        'error': 'Error al consultar API-Football',
                        'status_code': response.status_code
                    }), 500
                
                data = response.json()
                print(f"📦 API Response: {data.get('results', 0)} resultados")
                
                if data.get('response'):
                    for item in data['response'][:100]:
                        player_data = item.get('player', {})
                        players.append({
                            'player_id': player_data.get('id'),
                            'name': player_data.get('firstname', ''),
                            'lastname': player_data.get('lastname', ''),
                            'short_name': player_data.get('name', ''),
                            'nationality': player_data.get('nationality', ''),
                            'birth_date': player_data.get('birth', {}).get('date', ''),
                            'photo': player_data.get('photo', ''),
                            'source': 'api-football'
                        })
                    
                    print(f"✅ Encontrados {len(players)} jugadores en API-Football")
                else:
                    print(f"ℹ️ API-Football no devolvió resultados")
            
            except requests.Timeout:
                print(f"⏱️ Timeout al consultar API-Football")
                return jsonify({'error': 'Timeout al consultar API-Football'}), 504
            except Exception as e:
                print(f"❌ Error en API-Football: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Error en API-Football: {str(e)}'}), 500
        else:
            print(f"✅ Usando resultados de BD local, saltando API-Football")
        
        print(f"📊 Total jugadores a devolver: {len(players)}")
        return jsonify(players), 200
        
    except Exception as e:
        print(f"❌ Error general en search_players: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/api-football/teams/search', methods=['GET'])
def search_teams():
    """Busca equipos en API-Football por nombre"""
    try:
        search_query = request.args.get('search', '')
        
        if not search_query or len(search_query) < 3:
            return jsonify({'error': 'La búsqueda debe tener al menos 3 caracteres'}), 400
        
        # Buscar en API-Football
        api_url = f"{config.API_FOOTBALL_URL}/teams"
        params = {'search': search_query}
        
        response = requests.get(api_url, headers=API_FOOTBALL_HEADERS, params=params, timeout=10)
        
        if response.status_code != 200:
            return jsonify({'error': 'Error al consultar API-Football'}), 500
        
        data = response.json()
        
        # Formatear respuesta
        teams = []
        if data.get('response'):
            for item in data['response'][:100]:  # Limitar a 10 resultados
                team_data = item.get('team', {})
                teams.append({
                    'team_id': team_data.get('id'),
                    'name': team_data.get('name', ''),
                    'country': team_data.get('country', ''),
                    'logo': team_data.get('logo', '')
                })
        
        return jsonify(teams), 200
        
    except requests.Timeout:
        return jsonify({'error': 'Timeout al consultar API-Football'}), 504
    except Exception as e:
        print(f"Error en search_teams: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/api-football/leagues/search', methods=['GET'])
def search_leagues():
    """Busca ligas en API-Football por nombre"""
    try:
        search_query = request.args.get('search', '')
        
        if not search_query or len(search_query) < 3:
            return jsonify({'error': 'La búsqueda debe tener al menos 3 caracteres'}), 400
        
        # Buscar en API-Football
        api_url = f"{config.API_FOOTBALL_URL}/leagues"
        params = {'search': search_query}
        
        response = requests.get(api_url, headers=API_FOOTBALL_HEADERS, params=params, timeout=10)
        
        if response.status_code != 200:
            return jsonify({'error': 'Error al consultar API-Football'}), 500
        
        data = response.json()
        
        # Formatear respuesta
        leagues = []
        if data.get('response'):
            for item in data['response'][:100]:  # Limitar a 10 resultados
                league_data = item.get('league', {})
                leagues.append({
                    'league_id': league_data.get('id'),
                    'name': league_data.get('name', ''),
                    'country': league_data.get('country', ''),
                    'logo': league_data.get('logo', ''),
                    'type': league_data.get('type', '')
                })
        
        return jsonify(leagues), 200
        
    except requests.Timeout:
        return jsonify({'error': 'Timeout al consultar API-Football'}), 504
    except Exception as e:
        print(f"Error en search_leagues: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/api-football/fixtures/search', methods=['GET'])
def search_fixtures():
    """Busca fixtures por league, season con shooter_team y defender_team - combina resultados"""
    try:
        league_id = request.args.get('league')
        season = request.args.get('season')
        shooter_team_id = request.args.get('shooter_team')
        defender_team_id = request.args.get('defender_team')
        
        if not all([league_id, season, shooter_team_id, defender_team_id]):
            return jsonify({'error': 'Se requieren league, season, shooter_team y defender_team'}), 400
        
        all_fixtures = []
        seen_fixture_ids = set()
        
        # 1. BUSCAR CON SHOOTER_TEAM
        try:
            api_url = f"{config.API_FOOTBALL_URL}/fixtures"
            params = {
                'league': league_id,
                'season': season,
                'team': shooter_team_id
            }
            
            response = requests.get(api_url, headers=API_FOOTBALL_HEADERS, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response'):
                    for fixture_data in data['response']:
                        fixture_info = fixture_data.get('fixture', {})
                        fixture_id = fixture_info.get('id')
                        
                        if fixture_id and fixture_id not in seen_fixture_ids:
                            teams_info = fixture_data.get('teams', {})
                            
                            all_fixtures.append({
                                'fixture_id': fixture_id,
                                'date': fixture_info.get('date', ''),
                                'home_team': teams_info.get('home', {}).get('name', ''),
                                'home_team_id': teams_info.get('home', {}).get('id'),
                                'away_team': teams_info.get('away', {}).get('name', ''),
                                'away_team_id': teams_info.get('away', {}).get('id'),
                                'status': fixture_info.get('status', {}).get('long', ''),
                                'found_with': 'shooter_team'
                            })
                            seen_fixture_ids.add(fixture_id)
                    
                    print(f"✅ Encontrados {len(all_fixtures)} fixtures con shooter_team")
        except Exception as e:
            print(f"Error buscando con shooter_team: {e}")
        
        # 2. BUSCAR CON DEFENDER_TEAM
        try:
            api_url = f"{config.API_FOOTBALL_URL}/fixtures"
            params = {
                'league': league_id,
                'season': season,
                'team': defender_team_id
            }
            
            response = requests.get(api_url, headers=API_FOOTBALL_HEADERS, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response'):
                    initial_count = len(all_fixtures)
                    
                    for fixture_data in data['response']:
                        fixture_info = fixture_data.get('fixture', {})
                        fixture_id = fixture_info.get('id')
                        
                        if fixture_id and fixture_id not in seen_fixture_ids:
                            teams_info = fixture_data.get('teams', {})
                            
                            all_fixtures.append({
                                'fixture_id': fixture_id,
                                'date': fixture_info.get('date', ''),
                                'home_team': teams_info.get('home', {}).get('name', ''),
                                'home_team_id': teams_info.get('home', {}).get('id'),
                                'away_team': teams_info.get('away', {}).get('name', ''),
                                'away_team_id': teams_info.get('away', {}).get('id'),
                                'status': fixture_info.get('status', {}).get('long', ''),
                                'found_with': 'defender_team'
                            })
                            seen_fixture_ids.add(fixture_id)
                    
                    new_fixtures = len(all_fixtures) - initial_count
                    print(f"✅ Encontrados {new_fixtures} fixtures adicionales con defender_team")
        except Exception as e:
            print(f"Error buscando con defender_team: {e}")
        
        # Ordenar por fecha (más recientes primero)
        all_fixtures.sort(key=lambda x: x['date'], reverse=True)
        
        print(f"📊 Total de fixtures únicos encontrados: {len(all_fixtures)}")
        
        return jsonify({
            'fixtures': all_fixtures[:50],  # Limitar a 30 resultados
            'count': len(all_fixtures)
        }), 200
        
    except requests.Timeout:
        return jsonify({'error': 'Timeout al consultar API-Football'}), 504
    except Exception as e:
        print(f"Error en search_fixtures: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== ENDPOINTS DE UTILIDAD ====================

@app.route('/api/check-exists/<string:entity_type>/<int:entity_id>', methods=['GET'])
def check_entity_exists(entity_type, entity_id):
    """Verifica si una entidad ya existe en la base de datos"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if entity_type == 'player':
            cursor.execute("SELECT * FROM players WHERE player_id = %s", (entity_id,))
        elif entity_type == 'team':
            cursor.execute("SELECT * FROM teams WHERE team_id = %s", (entity_id,))
        elif entity_type == 'league':
            season = request.args.get('season')
            if not season:
                return jsonify({'error': 'Season required for league check'}), 400
            cursor.execute(
                "SELECT * FROM leagues WHERE league_id = %s AND season = %s", 
                (entity_id, season)
            )
        else:
            return jsonify({'error': 'Invalid entity type'}), 400
        
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'exists': result is not None,
            'data': dict(result) if result else None
        }), 200
        
    except Exception as e:
        print(f"Error en check_entity_exists: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API y la DB están funcionando"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        cursor.close()
        conn.close()
        
        return jsonify({
            'status': 'ok',
            'database': 'connected'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'database': 'disconnected',
            'message': str(e)
        }), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== ENDPOINTS DE VIDEO UPLOAD ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== ENDPOINTS DE VIDEO UPLOAD ====================

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """Sube un video para procesamiento temporal"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No se encontró el archivo de video'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido. Use: mp4, avi, mov, mkv'}), 400
        
        # Obtener penalty_id del form
        penalty_id = request.form.get('penalty_id')
        if not penalty_id:
            return jsonify({'error': 'Se requiere penalty_id'}), 400
        
        # Guardar archivo con nombre seguro
        filename = f"penalty_{penalty_id}_temp.mp4"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Obtener info del video
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"✅ Video subido: {filename}")
        print(f"📊 Info: {width}x{height}, {fps}FPS, {total_frames} frames, {duration:.2f}s")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            }
        }), 200
        
    except Exception as e:
        print(f"Error en upload_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/detect-players', methods=['POST'])
def detect_players():
    """Primera pasada: detecta y trackea jugadores usando YOLOv11"""
    try:
        data = request.json
        filepath = data.get('filepath')
        penalty_id = data.get('penalty_id')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        if not penalty_id:
            return jsonify({'error': 'Se requiere penalty_id'}), 400
        
        print(f"🔍 Iniciando detección de jugadores en: {filepath}")
        print(f"📝 Penalty ID: {penalty_id}")
        
        # Importar detector
        import sys
        import numpy as np
        sys.path.append(os.path.dirname(__file__))
        from detector import FootballPlayerDetector
        
        # Crear detector
        detector = FootballPlayerDetector(confidence_threshold=0.4)
        
        # Ruta para video procesado
        processed_video_path = os.path.join(UPLOAD_FOLDER, f"penalty_{penalty_id}_detected.mp4")
        
        # Procesar video (primera pasada) - AHORA GUARDA EL VIDEO
        detected_ids = detector.process_video_first_pass(
            video_path=filepath,
            output_path=processed_video_path,  # Guardar video con detecciones
            show_video=False   # No mostrar ventana
        )
        
        # Obtener estadísticas
        stats = detector.calculate_statistics()
        
        # Convertir numpy int32 a Python int para JSON serialization
        detected_ids_list = [int(id) for id in detected_ids]
        stats_serializable = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v 
                             for k, v in stats.items()}
        
        print(f"✅ Detección completada. IDs encontrados: {detected_ids_list}")
        print(f"📹 Video procesado guardado en: {processed_video_path}")
        
        return jsonify({
            'success': True,
            'detected_player_ids': detected_ids_list,
            'stats': stats_serializable,
            'processed_video_filename': os.path.basename(processed_video_path)
        }), 200
        
    except Exception as e:
        print(f"Error en detect_players: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/extract-postures', methods=['POST'])
def extract_postures():
    """Segunda pasada: extrae landmarks de jugadores seleccionados"""
    try:
        data = request.json
        filepath = data.get('filepath')
        selected_player_ids = data.get('player_ids', [])
        penalty_id = data.get('penalty_id')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        if not selected_player_ids:
            return jsonify({'error': 'Se requieren IDs de jugadores'}), 400
        
        print(f"🦴 Extrayendo landmarks de jugadores: {selected_player_ids}")
        
        # Importar detector
        import sys
        sys.path.append(os.path.dirname(__file__))
        from detector import FootballPlayerDetector
        
        # Crear detector
        detector = FootballPlayerDetector(confidence_threshold=0.4)
        
        # CSV temporal
        csv_path = os.path.join(UPLOAD_FOLDER, f"penalty_{penalty_id}_postures.csv")
        
        # Procesar video (segunda pasada)
        player_usage_stats, total_frames = detector.process_video_second_pass(
            video_path=filepath,
            selected_player_ids=selected_player_ids,
            csv_output_path=csv_path
        )
        
        print(f"✅ Extracción completada. CSV guardado en: {csv_path}")
        
        return jsonify({
            'success': True,
            'csv_path': csv_path,
            'player_usage_stats': player_usage_stats,
            'total_frames': total_frames
        }), 200
        
    except Exception as e:
        print(f"Error en extract_postures: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/temp/<filename>')
def serve_temp_video(filename):
    """Sirve un video temporal subido"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Video no encontrado'}), 404
        
        return send_file(
            filepath,
            mimetype='video/mp4',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"Error sirviendo video temporal: {e}")
        return jsonify({'error': str(e)}), 500
    
# ==================== ENDPOINTS DE INSERCIÓN ====================

@app.route('/api/insert/player', methods=['POST'])
def insert_player():
    """Inserta un jugador si no existe"""
    try:
        data = request.json
        player_id = data.get('player_id')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verificar si existe
        cursor.execute("SELECT player_id FROM players WHERE player_id = %s", (player_id,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("""
                INSERT INTO players (player_id, short_name, name, lastname, foot)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                player_id,
                data.get('short_name'),
                data.get('name'),
                data.get('lastname'),
                data.get('foot')
            ))
            conn.commit()
            print(f"✅ Jugador {player_id} insertado")
        else:
            print(f"ℹ️ Jugador {player_id} ya existe")
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'exists': exists is not None}), 200
        
    except Exception as e:
        print(f"Error en insert_player: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insert/team', methods=['POST'])
def insert_team():
    """Inserta un equipo si no existe"""
    try:
        data = request.json
        team_id = data.get('team_id')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verificar si existe
        cursor.execute("SELECT team_id FROM teams WHERE team_id = %s", (team_id,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("""
                INSERT INTO teams (team_id, name)
                VALUES (%s, %s)
            """, (team_id, data.get('name')))
            conn.commit()
            print(f"✅ Equipo {team_id} insertado")
        else:
            print(f"ℹ️ Equipo {team_id} ya existe")
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'exists': exists is not None}), 200
        
    except Exception as e:
        print(f"Error en insert_team: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insert/league', methods=['POST'])
def insert_league():
    """Inserta una liga si no existe"""
    try:
        data = request.json
        league_id = data.get('league_id')
        season = data.get('season')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verificar si existe
        cursor.execute(
            "SELECT league_id FROM leagues WHERE league_id = %s AND season = %s",
            (league_id, season)
        )
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("""
                INSERT INTO leagues (league_id, season, name)
                VALUES (%s, %s, %s)
            """, (league_id, season, data.get('name')))
            conn.commit()
            print(f"✅ Liga {league_id} temporada {season} insertada")
        else:
            print(f"ℹ️ Liga {league_id} temporada {season} ya existe")
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'exists': exists is not None}), 200
        
    except Exception as e:
        print(f"Error en insert_league: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insert/penalty', methods=['POST'])
def insert_penalty():
    """Inserta un penal"""
    try:
        data = request.json
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO penalties (
                penalty_id, fixture_id, league_id, season, event,
                minute, extra_minute, shooter_team_id, defender_team_id,
                player_id, condition, penalty_shootout, height, side
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data.get('penalty_id'),
            data.get('fixture_id'),
            data.get('league_id'),
            data.get('season'),
            data.get('event'),
            data.get('minute'),
            data.get('extra_minute'),
            data.get('shooter_team_id'),
            data.get('defender_team_id'),
            data.get('player_id'),
            data.get('condition'),
            data.get('penalty_shootout'),
            data.get('height'),
            data.get('side')
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Penal {data.get('penalty_id')} insertado")
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        print(f"Error en insert_penalty: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insert/postures', methods=['POST'])
def insert_postures():
    """Inserta posturas desde un archivo CSV"""
    try:
        data = request.json
        penalty_id = data.get('penalty_id')
        csv_path = data.get('csv_path')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Archivo CSV no encontrado'}), 404
        
        import pandas as pd
        import numpy as np
        
        # Leer CSV
        df = pd.read_csv(csv_path)
        
        # CRÍTICO: Reemplazar NaN con None (null en SQL)
        df = df.replace({np.nan: None})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insertar cada frame
        inserted_count = 0
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO postures (
                    penalty_id, frame,
                    nose_x, nose_y, nose_confidence,
                    left_eye_x, left_eye_y, left_eye_confidence,
                    right_eye_x, right_eye_y, right_eye_confidence,
                    left_ear_x, left_ear_y, left_ear_confidence,
                    right_ear_x, right_ear_y, right_ear_confidence,
                    left_shoulder_x, left_shoulder_y, left_shoulder_confidence,
                    right_shoulder_x, right_shoulder_y, right_shoulder_confidence,
                    left_elbow_x, left_elbow_y, left_elbow_confidence,
                    right_elbow_x, right_elbow_y, right_elbow_confidence,
                    left_wrist_x, left_wrist_y, left_wrist_confidence,
                    right_wrist_x, right_wrist_y, right_wrist_confidence,
                    left_hip_x, left_hip_y, left_hip_confidence,
                    right_hip_x, right_hip_y, right_hip_confidence,
                    left_knee_x, left_knee_y, left_knee_confidence,
                    right_knee_x, right_knee_y, right_knee_confidence,
                    left_ankle_x, left_ankle_y, left_ankle_confidence,
                    right_ankle_x, right_ankle_y, right_ankle_confidence
                )
                VALUES (
                    %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
            """, (
                penalty_id, int(row['frame']),
                row['nose_x'], row['nose_y'], row['nose_confidence'],
                row['left_eye_x'], row['left_eye_y'], row['left_eye_confidence'],
                row['right_eye_x'], row['right_eye_y'], row['right_eye_confidence'],
                row['left_ear_x'], row['left_ear_y'], row['left_ear_confidence'],
                row['right_ear_x'], row['right_ear_y'], row['right_ear_confidence'],
                row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_confidence'],
                row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_confidence'],
                row['left_elbow_x'], row['left_elbow_y'], row['left_elbow_confidence'],
                row['right_elbow_x'], row['right_elbow_y'], row['right_elbow_confidence'],
                row['left_wrist_x'], row['left_wrist_y'], row['left_wrist_confidence'],
                row['right_wrist_x'], row['right_wrist_y'], row['right_wrist_confidence'],
                row['left_hip_x'], row['left_hip_y'], row['left_hip_confidence'],
                row['right_hip_x'], row['right_hip_y'], row['right_hip_confidence'],
                row['left_knee_x'], row['left_knee_y'], row['left_knee_confidence'],
                row['right_knee_x'], row['right_knee_y'], row['right_knee_confidence'],
                row['left_ankle_x'], row['left_ankle_y'], row['left_ankle_confidence'],
                row['right_ankle_x'], row['right_ankle_y'], row['right_ankle_confidence']
            ))
            inserted_count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ {inserted_count} posturas insertadas para penal {penalty_id}")
        
        return jsonify({'success': True, 'frames_inserted': inserted_count}), 200
        
    except Exception as e:
        print(f"Error en insert_postures: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/video-to-s3', methods=['POST'])
def upload_video_to_s3():
    """Sube el video procesado a S3"""
    try:
        data = request.json
        penalty_id = data.get('penalty_id')
        original_video_path = data.get('original_video_path')
        
        if not os.path.exists(original_video_path):
            return jsonify({'error': 'Video no encontrado'}), 404
        
        # Nombre del archivo en S3
        s3_key = f"{penalty_id}.mp4"
        
        print(f"📤 Subiendo video a S3: {s3_key}")
        
        # Subir a S3
        with open(original_video_path, 'rb') as video_file:
            s3_client.upload_fileobj(
                video_file,
                config.S3_BUCKET_NAME,
                s3_key,
                ExtraArgs={
                    'ContentType': 'video/mp4',
                    'ACL': 'private'
                }
            )
        
        print(f"✅ Video subido exitosamente a S3: {s3_key}")
        
        # Limpiar archivos temporales
        try:
            if os.path.exists(original_video_path):
                os.remove(original_video_path)
                print(f"🗑️ Archivo temporal eliminado: {original_video_path}")
        except Exception as e:
            print(f"⚠️ Error al eliminar archivo temporal: {e}")
        
        return jsonify({
            'success': True,
            's3_key': s3_key,
            'bucket': config.S3_BUCKET_NAME
        }), 200
        
    except ClientError as e:
        print(f"Error S3: {e}")
        return jsonify({'error': f'Error al subir a S3: {str(e)}'}), 500
    except Exception as e:
        print(f"Error en upload_video_to_s3: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== ENDPOINTS DE PREDICCIÓN ====================

@app.route('/api/prediction/upload-video', methods=['POST'])
def prediction_upload_video():
    """Sube un video temporal para predicción"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No se encontró el archivo de video'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido. Use: mp4, avi, mov, mkv'}), 400
        
        # Generar nombre único temporal
        import uuid
        temp_id = str(uuid.uuid4())
        filename = f"prediction_{temp_id}_temp.mp4"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Obtener info del video
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"✅ Video de predicción subido: {filename}")
        print(f"📊 Info: {width}x{height}, {fps}FPS, {total_frames} frames, {duration:.2f}s")
        
        return jsonify({
            'success': True,
            'temp_id': temp_id,
            'filename': filename,
            'filepath': filepath,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            }
        }), 200
        
    except Exception as e:
        print(f"Error en prediction_upload_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction/detect-players', methods=['POST'])
def prediction_detect_players():
    """Detecta jugadores en video de predicción usando YOLOv11"""
    try:
        data = request.json
        filepath = data.get('filepath')
        temp_id = data.get('temp_id')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        print(f"🔍 Iniciando detección de jugadores en: {filepath}")
        
        # Importar detector
        import sys
        sys.path.append(os.path.dirname(__file__))
        from detector import FootballPlayerDetector
        
        # Crear detector
        detector = FootballPlayerDetector(confidence_threshold=0.4)
        
        # Ruta para video procesado
        processed_video_path = os.path.join(UPLOAD_FOLDER, f"prediction_{temp_id}_detected.mp4")
        
        # Procesar video (primera pasada)
        detected_ids = detector.process_video_first_pass(
            video_path=filepath,
            output_path=processed_video_path,
            show_video=False
        )
        
        # Obtener estadísticas
        stats = detector.calculate_statistics()
        
        # Convertir numpy types a Python natives
        detected_ids_list = [int(id) for id in detected_ids]
        stats_serializable = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v 
                             for k, v in stats.items()}
        
        print(f"✅ Detección completada. IDs encontrados: {detected_ids_list}")
        
        return jsonify({
            'success': True,
            'detected_player_ids': detected_ids_list,
            'stats': stats_serializable,
            'processed_video_filename': os.path.basename(processed_video_path)
        }), 200
        
    except Exception as e:
        print(f"Error en prediction_detect_players: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction/extract-and-predict', methods=['POST'])
def prediction_extract_and_predict():
    """Extrae posturas y ejecuta predicción ML"""
    try:
        data = request.json
        filepath = data.get('filepath')
        temp_id = data.get('temp_id')
        selected_player_ids = data.get('player_ids', [])
        player_foot = data.get('player_foot')  # 'L' o 'R'
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        if not selected_player_ids:
            return jsonify({'error': 'Se requieren IDs de jugadores'}), 400
        
        if not player_foot or player_foot not in ['L', 'R']:
            return jsonify({'error': 'Se requiere pie del jugador (L o R)'}), 400
        
        print(f"🦴 Extrayendo landmarks de jugadores: {selected_player_ids}")
        print(f"👟 Pie del pateador: {player_foot}")
        
        # 1. EXTRAER POSTURAS
        import sys
        sys.path.append(os.path.dirname(__file__))
        from detector import FootballPlayerDetector
        
        detector = FootballPlayerDetector(confidence_threshold=0.4)
        
        # CSV temporal
        csv_path = os.path.join(UPLOAD_FOLDER, f"prediction_{temp_id}_postures.csv")
        
        # Procesar video (segunda pasada)
        player_usage_stats, total_frames = detector.process_video_second_pass(
            video_path=filepath,
            selected_player_ids=selected_player_ids,
            csv_output_path=csv_path
        )
        
        print(f"✅ Extracción completada. CSV guardado en: {csv_path}")
        
        # 2. CARGAR CSV Y PREPARAR DATOS
        df = pd.read_csv(csv_path)
        
        # Renombrar columnas al formato esperado por el modelo
        column_mapping = {
            'nose_x': 'NOSE_X', 'nose_y': 'NOSE_Y', 'nose_confidence': 'NOSE_CONFIDENCE',
            'left_eye_x': 'LEFT_EYE_X', 'left_eye_y': 'LEFT_EYE_Y', 'left_eye_confidence': 'LEFT_EYE_CONFIDENCE',
            'right_eye_x': 'RIGHT_EYE_X', 'right_eye_y': 'RIGHT_EYE_Y', 'right_eye_confidence': 'RIGHT_EYE_CONFIDENCE',
            'left_ear_x': 'LEFT_EAR_X', 'left_ear_y': 'LEFT_EAR_Y', 'left_ear_confidence': 'LEFT_EAR_CONFIDENCE',
            'right_ear_x': 'RIGHT_EAR_X', 'right_ear_y': 'RIGHT_EAR_Y', 'right_ear_confidence': 'RIGHT_EAR_CONFIDENCE',
            'left_shoulder_x': 'LEFT_SHOULDER_X', 'left_shoulder_y': 'LEFT_SHOULDER_Y', 'left_shoulder_confidence': 'LEFT_SHOULDER_CONFIDENCE',
            'right_shoulder_x': 'RIGHT_SHOULDER_X', 'right_shoulder_y': 'RIGHT_SHOULDER_Y', 'right_shoulder_confidence': 'RIGHT_SHOULDER_CONFIDENCE',
            'left_elbow_x': 'LEFT_ELBOW_X', 'left_elbow_y': 'LEFT_ELBOW_Y', 'left_elbow_confidence': 'LEFT_ELBOW_CONFIDENCE',
            'right_elbow_x': 'RIGHT_ELBOW_X', 'right_elbow_y': 'RIGHT_ELBOW_Y', 'right_elbow_confidence': 'RIGHT_ELBOW_CONFIDENCE',
            'left_wrist_x': 'LEFT_WRIST_X', 'left_wrist_y': 'LEFT_WRIST_Y', 'left_wrist_confidence': 'LEFT_WRIST_CONFIDENCE',
            'right_wrist_x': 'RIGHT_WRIST_X', 'right_wrist_y': 'RIGHT_WRIST_Y', 'right_wrist_confidence': 'RIGHT_WRIST_CONFIDENCE',
            'left_hip_x': 'LEFT_HIP_X', 'left_hip_y': 'LEFT_HIP_Y', 'left_hip_confidence': 'LEFT_HIP_CONFIDENCE',
            'right_hip_x': 'RIGHT_HIP_X', 'right_hip_y': 'RIGHT_HIP_Y', 'right_hip_confidence': 'RIGHT_HIP_CONFIDENCE',
            'left_knee_x': 'LEFT_KNEE_X', 'left_knee_y': 'LEFT_KNEE_Y', 'left_knee_confidence': 'LEFT_KNEE_CONFIDENCE',
            'right_knee_x': 'RIGHT_KNEE_X', 'right_knee_y': 'RIGHT_KNEE_Y', 'right_knee_confidence': 'RIGHT_KNEE_CONFIDENCE',
            'left_ankle_x': 'LEFT_ANKLE_X', 'left_ankle_y': 'LEFT_ANKLE_Y', 'left_ankle_confidence': 'LEFT_ANKLE_CONFIDENCE',
            'right_ankle_x': 'RIGHT_ANKLE_X', 'right_ankle_y': 'RIGHT_ANKLE_Y', 'right_ankle_confidence': 'RIGHT_ANKLE_CONFIDENCE',
            'frame': 'FRAME'
        }
        df = df.rename(columns=column_mapping)
        df['PLAYER_FOOT'] = player_foot
        
        # 3. FEATURE ENGINEERING
        print("🔧 Aplicando feature engineering...")
        engineered_features = df.apply(engineer_features_per_frame, axis=1)
        df_with_features = pd.concat([df, engineered_features], axis=1)
        
        # 4. CARGAR MODELOS
        print("📦 Cargando modelos ML...")
        models_path = os.path.join(os.path.dirname(__file__), 'models')
        
        model_height = joblib.load(os.path.join(models_path, 'model_height.joblib'))
        model_side = joblib.load(os.path.join(models_path, 'model_side.joblib'))
        le_height = joblib.load(os.path.join(models_path, 'label_encoder_height.joblib'))
        le_side = joblib.load(os.path.join(models_path, 'label_encoder_side.joblib'))
        le_foot = joblib.load(os.path.join(models_path, 'label_encoder_foot.joblib'))
        
        with open(os.path.join(models_path, 'feature_columns.json'), 'r') as f:
            feature_columns = json.load(f)
        
        # ENCODEAR PLAYER_FOOT CON MANEJO ROBUSTO
        try:
            print(f"👟 Procesando pie del jugador: {player_foot}")
            print(f"🔍 Clases del encoder: {le_foot.classes_}")
            print(f"🔍 Tipo de clases: {type(le_foot.classes_[0])}")
            
            # Verificar si el encoder usa strings o números
            first_class = le_foot.classes_[0]
            
            if isinstance(first_class, str):
                # El encoder espera strings directamente
                print(f"✅ Encoder usa strings, transformando directamente...")
                df_with_features['PLAYER_FOOT_ENCODED'] = le_foot.transform(df_with_features['PLAYER_FOOT'])
                print(f"✅ Transformación directa exitosa: {player_foot} -> {df_with_features['PLAYER_FOOT_ENCODED'].iloc[0]}")
                
            elif isinstance(first_class, (int, np.integer)):
                # El encoder usa números - necesitamos mapear manualmente
                print(f"⚠️ Encoder usa números, aplicando mapeo manual...")
                
                # Mapeo estándar: L=0, R=1, B=2 (si existe)
                foot_mapping = {'L': 0, 'R': 1}
                if len(le_foot.classes_) > 2:
                    foot_mapping['B'] = 2
                
                print(f"📋 Mapeo aplicado: {foot_mapping}")
                
                encoded_value = foot_mapping.get(player_foot)
                if encoded_value is None:
                    raise ValueError(f"Pie '{player_foot}' no está en el mapeo: {foot_mapping}")
                
                df_with_features['PLAYER_FOOT_ENCODED'] = encoded_value
                print(f"✅ PLAYER_FOOT_ENCODED = {encoded_value}")
                
            else:
                # Tipo desconocido, usar mapeo por defecto
                print(f"⚠️ Tipo de clase desconocido: {type(first_class)}, usando mapeo por defecto")
                foot_mapping = {'L': 0, 'R': 1, 'B': 2}
                encoded_value = foot_mapping.get(player_foot, 0)
                df_with_features['PLAYER_FOOT_ENCODED'] = encoded_value
                print(f"✅ Usando valor por defecto: {encoded_value}")
        
        except Exception as e:
            print(f"❌ Error crítico al encodear PLAYER_FOOT: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback seguro: asumir L=0, R=1
            print("⚠️ Aplicando fallback seguro: L=0, R=1")
            fallback_map = {'L': 0, 'R': 1, 'B': 2}
            encoded_value = fallback_map.get(player_foot, 0)
            df_with_features['PLAYER_FOOT_ENCODED'] = encoded_value
            print(f"✅ Fallback aplicado: {player_foot} -> {encoded_value}")
        
        # 5. PREPARAR DATOS PARA PREDICCIÓN
        print(f"📋 Preparando features: {len(feature_columns)} columnas esperadas")
        X_pred = df_with_features[feature_columns]
        X_pred = X_pred.replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"✅ Dataset preparado: {X_pred.shape}")
        
        # 6. PREDICCIONES
        print("🎯 Realizando predicciones...")
        y_height_pred_encoded = model_height.predict(X_pred)
        y_height_pred = le_height.inverse_transform(y_height_pred_encoded)
        
        y_side_pred_encoded = model_side.predict(X_pred)
        y_side_pred = le_side.inverse_transform(y_side_pred_encoded)
        
        y_height_proba = model_height.predict_proba(X_pred)
        y_side_proba = model_side.predict_proba(X_pred)
        
        height_confidence = y_height_proba.max(axis=1)
        side_confidence = y_side_proba.max(axis=1)
        
        # 7. ANÁLISIS DE RESULTADOS
        results_frames = pd.DataFrame({
            'FRAME': df_with_features['FRAME'],
            'PREDICTED_HEIGHT': y_height_pred,
            'HEIGHT_CONFIDENCE': height_confidence,
            'PREDICTED_SIDE': y_side_pred,
            'SIDE_CONFIDENCE': side_confidence
        })
        
        # Probabilidades promedio por clase
        height_classes = le_height.classes_
        side_classes = le_side.classes_
        
        height_probabilities = {}
        for idx, cls in enumerate(height_classes):
            height_probabilities[cls] = float(y_height_proba[:, idx].mean())
        
        side_probabilities = {}
        for idx, cls in enumerate(side_classes):
            side_probabilities[cls] = float(y_side_proba[:, idx].mean())
        
        # Distribución de frecuencias
        total_frames_count = len(results_frames)
        
        height_distribution = {}
        for cls in height_classes:
            count = int((results_frames['PREDICTED_HEIGHT'] == cls).sum())
            height_distribution[cls] = {
                'count': count,
                'percentage': float(count / total_frames_count * 100)
            }
        
        side_distribution = {}
        for cls in side_classes:
            count = int((results_frames['PREDICTED_SIDE'] == cls).sum())
            side_distribution[cls] = {
                'count': count,
                'percentage': float(count / total_frames_count * 100)
            }
        
        # Predicción final (votación mayoritaria)
        final_height = results_frames['PREDICTED_HEIGHT'].mode()[0]
        final_side = results_frames['PREDICTED_SIDE'].mode()[0]
        
        final_height_confidence = float(results_frames[results_frames['PREDICTED_HEIGHT'] == final_height]['HEIGHT_CONFIDENCE'].mean())
        final_side_confidence = float(results_frames[results_frames['PREDICTED_SIDE'] == final_side]['SIDE_CONFIDENCE'].mean())
        
        final_height_votes = int((results_frames['PREDICTED_HEIGHT'] == final_height).sum())
        final_side_votes = int((results_frames['PREDICTED_SIDE'] == final_side).sum())
        
        # Confianza global
        global_confidence = float((final_height_confidence + final_side_confidence) / 2)
        
        # Consistencia
        height_consistency = float(final_height_votes / total_frames_count)
        side_consistency = float(final_side_votes / total_frames_count)
        
        # Predicciones por frame (primeros 100 para no saturar)
        frame_predictions = results_frames.head(100).to_dict('records')
        
        # Limpiar archivos temporales
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"🗑️ Eliminado: {filepath}")
            if os.path.exists(csv_path):
                os.remove(csv_path)
                print(f"🗑️ Eliminado: {csv_path}")
            processed_video = os.path.join(UPLOAD_FOLDER, f"prediction_{temp_id}_detected.mp4")
            if os.path.exists(processed_video):
                os.remove(processed_video)
                print(f"🗑️ Eliminado: {processed_video}")
        except Exception as e:
            print(f"⚠️ Error al eliminar archivos temporales: {e}")
        
        print("✅ Predicción completada exitosamente")
        
        return jsonify({
            'success': True,
            'total_frames': int(total_frames_count),
            'player_foot': player_foot,
            'height_probabilities': height_probabilities,
            'side_probabilities': side_probabilities,
            'height_distribution': height_distribution,
            'side_distribution': side_distribution,
            'final_prediction': {
                'height': final_height,
                'height_confidence': final_height_confidence,
                'height_votes': final_height_votes,
                'height_percentage': float(final_height_votes / total_frames_count * 100),
                'side': final_side,
                'side_confidence': final_side_confidence,
                'side_votes': final_side_votes,
                'side_percentage': float(final_side_votes / total_frames_count * 100),
                'global_confidence': global_confidence
            },
            'consistency': {
                'height': height_consistency,
                'side': side_consistency
            },
            'frame_predictions': frame_predictions
        }), 200
        
    except Exception as e:
        print(f"Error en prediction_extract_and_predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Funciones auxiliares para feature engineering (del predict.py)
def calculate_angle(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    vector1 = np.array([p1_x - p2_x, p1_y - p2_y])
    vector2 = np.array([p3_x - p2_x, p3_y - p2_y])
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def engineer_features_per_frame(frame_df):
    features = {}
    
    # Ángulos
    features['left_elbow_angle'] = calculate_angle(
        frame_df['LEFT_SHOULDER_X'], frame_df['LEFT_SHOULDER_Y'],
        frame_df['LEFT_ELBOW_X'], frame_df['LEFT_ELBOW_Y'],
        frame_df['LEFT_WRIST_X'], frame_df['LEFT_WRIST_Y']
    )
    features['right_elbow_angle'] = calculate_angle(
        frame_df['RIGHT_SHOULDER_X'], frame_df['RIGHT_SHOULDER_Y'],
        frame_df['RIGHT_ELBOW_X'], frame_df['RIGHT_ELBOW_Y'],
        frame_df['RIGHT_WRIST_X'], frame_df['RIGHT_WRIST_Y']
    )
    features['left_knee_angle'] = calculate_angle(
        frame_df['LEFT_HIP_X'], frame_df['LEFT_HIP_Y'],
        frame_df['LEFT_KNEE_X'], frame_df['LEFT_KNEE_Y'],
        frame_df['LEFT_ANKLE_X'], frame_df['LEFT_ANKLE_Y']
    )
    features['right_knee_angle'] = calculate_angle(
        frame_df['RIGHT_HIP_X'], frame_df['RIGHT_HIP_Y'],
        frame_df['RIGHT_KNEE_X'], frame_df['RIGHT_KNEE_Y'],
        frame_df['RIGHT_ANKLE_X'], frame_df['RIGHT_ANKLE_Y']
    )
    features['hip_angle'] = calculate_angle(
        frame_df['LEFT_SHOULDER_X'], frame_df['LEFT_SHOULDER_Y'],
        frame_df['LEFT_HIP_X'], frame_df['LEFT_HIP_Y'],
        frame_df['LEFT_KNEE_X'], frame_df['LEFT_KNEE_Y']
    )
    features['left_shoulder_angle'] = calculate_angle(
        frame_df['LEFT_ELBOW_X'], frame_df['LEFT_ELBOW_Y'],
        frame_df['LEFT_SHOULDER_X'], frame_df['LEFT_SHOULDER_Y'],
        frame_df['LEFT_HIP_X'], frame_df['LEFT_HIP_Y']
    )
    features['right_shoulder_angle'] = calculate_angle(
        frame_df['RIGHT_ELBOW_X'], frame_df['RIGHT_ELBOW_Y'],
        frame_df['RIGHT_SHOULDER_X'], frame_df['RIGHT_SHOULDER_Y'],
        frame_df['RIGHT_HIP_X'], frame_df['RIGHT_HIP_Y']
    )
    
    # Distancias
    features['shoulder_width'] = calculate_distance(
        frame_df['LEFT_SHOULDER_X'], frame_df['LEFT_SHOULDER_Y'],
        frame_df['RIGHT_SHOULDER_X'], frame_df['RIGHT_SHOULDER_Y']
    )
    features['hip_width'] = calculate_distance(
        frame_df['LEFT_HIP_X'], frame_df['LEFT_HIP_Y'],
        frame_df['RIGHT_HIP_X'], frame_df['RIGHT_HIP_Y']
    )
    features['feet_distance'] = calculate_distance(
        frame_df['LEFT_ANKLE_X'], frame_df['LEFT_ANKLE_Y'],
        frame_df['RIGHT_ANKLE_X'], frame_df['RIGHT_ANKLE_Y']
    )
    
    avg_ankle_y = (frame_df['LEFT_ANKLE_Y'] + frame_df['RIGHT_ANKLE_Y']) / 2
    features['body_height'] = abs(frame_df['NOSE_Y'] - avg_ankle_y)
    
    features['left_arm_length'] = (
        calculate_distance(frame_df['LEFT_SHOULDER_X'], frame_df['LEFT_SHOULDER_Y'],
                         frame_df['LEFT_ELBOW_X'], frame_df['LEFT_ELBOW_Y']) +
        calculate_distance(frame_df['LEFT_ELBOW_X'], frame_df['LEFT_ELBOW_Y'],
                         frame_df['LEFT_WRIST_X'], frame_df['LEFT_WRIST_Y'])
    )
    features['right_arm_length'] = (
        calculate_distance(frame_df['RIGHT_SHOULDER_X'], frame_df['RIGHT_SHOULDER_Y'],
                         frame_df['RIGHT_ELBOW_X'], frame_df['RIGHT_ELBOW_Y']) +
        calculate_distance(frame_df['RIGHT_ELBOW_X'], frame_df['RIGHT_ELBOW_Y'],
                         frame_df['RIGHT_WRIST_X'], frame_df['RIGHT_WRIST_Y'])
    )
    features['left_leg_length'] = (
        calculate_distance(frame_df['LEFT_HIP_X'], frame_df['LEFT_HIP_Y'],
                         frame_df['LEFT_KNEE_X'], frame_df['LEFT_KNEE_Y']) +
        calculate_distance(frame_df['LEFT_KNEE_X'], frame_df['LEFT_KNEE_Y'],
                         frame_df['LEFT_ANKLE_X'], frame_df['LEFT_ANKLE_Y'])
    )
    features['right_leg_length'] = (
        calculate_distance(frame_df['RIGHT_HIP_X'], frame_df['RIGHT_HIP_Y'],
                         frame_df['RIGHT_KNEE_X'], frame_df['RIGHT_KNEE_Y']) +
        calculate_distance(frame_df['RIGHT_KNEE_X'], frame_df['RIGHT_KNEE_Y'],
                         frame_df['RIGHT_ANKLE_X'], frame_df['RIGHT_ANKLE_Y'])
    )
    
    # Posiciones relativas
    center_x = (frame_df['LEFT_HIP_X'] + frame_df['RIGHT_HIP_X']) / 2
    center_y = (frame_df['LEFT_HIP_Y'] + frame_df['RIGHT_HIP_Y']) / 2
    features['nose_deviation_x'] = frame_df['NOSE_X'] - center_x
    features['nose_deviation_y'] = frame_df['NOSE_Y'] - center_y
    features['left_wrist_deviation_x'] = frame_df['LEFT_WRIST_X'] - center_x
    features['right_wrist_deviation_x'] = frame_df['RIGHT_WRIST_X'] - center_x
    
    # Ratios
    features['hip_shoulder_ratio'] = features['hip_width'] / features['shoulder_width'] if features['shoulder_width'] > 0 else 0
    
    # Confianza
    confidence_cols = [col for col in frame_df.index if 'CONFIDENCE' in col]
    features['avg_confidence'] = frame_df[confidence_cols].mean()
    features['min_confidence'] = frame_df[confidence_cols].min()
    
    return pd.Series(features)

@app.route('/api/players/<int:player_id>/penalties', methods=['GET'])
def get_player_penalties(player_id):
    """Obtiene todos los penales de un jugador específico"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                p.penalty_id,
                p.fixture_id,
                p.minute,
                p.extra_minute,
                p.event,
                p.condition,
                p.penalty_shootout,
                p.height,
                p.side,
                l.name as league_name,
                l.season,
                st.name as shooter_team_name,
                dt.name as defender_team_name
            FROM penalties p
            LEFT JOIN leagues l ON p.league_id = l.league_id AND p.season = l.season
            LEFT JOIN teams st ON p.shooter_team_id = st.team_id
            LEFT JOIN teams dt ON p.defender_team_id = dt.team_id
            WHERE p.player_id = %s
            ORDER BY l.season DESC, p.penalty_id DESC
        """
        
        cursor.execute(query, (player_id,))
        penalties = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify(penalties), 200
        
    except Exception as e:
        print(f"Error en get_player_penalties: {e}")
        return jsonify({
            'error': 'Error al obtener penales del jugador',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)