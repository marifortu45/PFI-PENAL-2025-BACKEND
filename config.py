import os
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Configuración de la base de datos RDS PostgreSQL
DB_HOST = os.getenv('DB_HOST', 'your-rds-endpoint.rds.amazonaws.com')
DB_NAME = os.getenv('DB_NAME', 'penal_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_password')
DB_PORT = os.getenv('DB_PORT', '5432')

# Otras configuraciones
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Configuración de AWS
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Si no hay credenciales en .env, boto3 usará las del sistema (~/.aws/credentials)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'video-clips-pfi-penal')

# API-Football Configuration
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY', '')
API_FOOTBALL_URL = os.getenv('API_FOOTBALL_URL', 'https://v3.football.api-sports.io')