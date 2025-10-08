"""
V2 Modificada para YOLOv11 - Nombres de archivo basados en video + Selección múltiple
pip install ultralytics>=8.0.196
python .\detector_v2_yolov11.py .\videos_limpios\1.mp4
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import argparse
import os
import pandas as pd
from scipy.spatial.distance import cdist

class PlayerTracker:
    def __init__(self, max_distance=100, max_frames_lost=10):
        """
        Tracker para mantener IDs consistentes de jugadores
        
        Args:
            max_distance: Distancia máxima para asociar detecciones (píxeles)
            max_frames_lost: Frames máximos sin detección antes de eliminar tracker
        """
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.tracks = {}  # {track_id: {'center': (x, y), 'frames_lost': int, 'bbox': (x1,y1,x2,y2)}}
        self.next_id = 1
        
    def update(self, detections):
        """
        Actualiza los tracks con nuevas detecciones
        
        Args:
            detections: Lista de detecciones [(x1, y1, x2, y2, conf), ...]
        
        Returns:
            Lista de tracks [(track_id, x1, y1, x2, y2, conf), ...]
        """
        if not detections:
            # Incrementar frames perdidos para todos los tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['frames_lost'] += 1
                if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                    del self.tracks[track_id]
            return []
        
        # Calcular centros de las detecciones actuales
        current_centers = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_centers.append((center_x, center_y))
        
        current_centers = np.array(current_centers)
        
        # Obtener centros de tracks existentes
        if self.tracks:
            track_ids = list(self.tracks.keys())
            track_centers = np.array([self.tracks[tid]['center'] for tid in track_ids])
            
            # Calcular matriz de distancias
            distances = cdist(current_centers, track_centers)
            
            # Asignación húngara simplificada (greedy)
            assigned_detections = set()
            assigned_tracks = set()
            assignments = []
            
            # Ordenar por distancia y asignar
            distance_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)
            
            for det_idx, track_idx in zip(distance_indices[0], distance_indices[1]):
                if (det_idx not in assigned_detections and 
                    track_idx not in assigned_tracks and 
                    distances[det_idx, track_idx] < self.max_distance):
                    
                    assignments.append((det_idx, track_ids[track_idx]))
                    assigned_detections.add(det_idx)
                    assigned_tracks.add(track_idx)
        else:
            track_ids = []
            assignments = []
            assigned_detections = set()
            assigned_tracks = set()
        
        # Actualizar tracks existentes
        for det_idx, track_id in assignments:
            detection = detections[det_idx]
            x1, y1, x2, y2 = detection[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            self.tracks[track_id]['center'] = (center_x, center_y)
            self.tracks[track_id]['bbox'] = (x1, y1, x2, y2)
            self.tracks[track_id]['frames_lost'] = 0
            self.tracks[track_id]['conf'] = detection[4]
        
        # Crear nuevos tracks para detecciones no asignadas
        for det_idx, detection in enumerate(detections):
            if det_idx not in assigned_detections:
                x1, y1, x2, y2 = detection[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                self.tracks[self.next_id] = {
                    'center': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'frames_lost': 0,
                    'conf': detection[4]
                }
                assignments.append((det_idx, self.next_id))
                self.next_id += 1
        
        # Incrementar frames perdidos para tracks no asignados
        for track_id in self.tracks:
            if track_id not in [tid for _, tid in assignments]:
                self.tracks[track_id]['frames_lost'] += 1
        
        # Eliminar tracks perdidos
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                del self.tracks[track_id]
        
        # Retornar tracks activos
        result = []
        for _, track_id in assignments:
            track = self.tracks[track_id]
            bbox = track['bbox']
            result.append((track_id, bbox[0], bbox[1], bbox[2], bbox[3], track['conf']))
        
        return result

class FootballPlayerDetector:
    def __init__(self, model_path=None, confidence_threshold=0.4):
        """
        Inicializa el detector de jugadores de fútbol con tracking usando YOLOv11
        
        Args:
            model_path: Ruta al modelo YOLO (si es None, usa YOLOv11 pre-entrenado)
            confidence_threshold: Umbral de confianza para las detecciones
        """
        self.confidence_threshold = confidence_threshold
        
        # Cargar modelo YOLOv11 para detección
        if model_path and os.path.exists(model_path):
            print(f"🔧 Cargando modelo personalizado: {model_path}")
            self.detection_model = YOLO(model_path)
        else:
            print("🔧 Cargando YOLOv11n para detección de personas...")
            self.detection_model = YOLO('yolo11n.pt')  # YOLOv11 nano
        
        # Cargar modelo YOLOv11 para pose estimation (17 keypoints)
        print("🦴 Cargando YOLOv11n-pose para análisis de pose...")
        self.pose_model = YOLO('yolo11n-pose.pt')  # YOLOv11 pose nano
        
        # Verificar que los modelos se cargaron correctamente
        print(f"✅ Modelo de detección cargado: {self.detection_model.model_name if hasattr(self.detection_model, 'model_name') else 'YOLOv11'}")
        print(f"✅ Modelo de pose cargado: {self.pose_model.model_name if hasattr(self.pose_model, 'model_name') else 'YOLOv11-pose'}")
        
        # Inicializar tracker
        self.tracker = PlayerTracker(max_distance=80, max_frames_lost=15)
        
        # Colores únicos para cada ID
        np.random.seed(42)
        self.colors = {}
        
        # Contador de jugadores por frame
        self.player_counts = []
        
        # Nombres de los 17 keypoints de COCO pose
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Para almacenar los IDs detectados durante el primer análisis
        self.detected_player_ids = set()
        
    def generate_color_for_id(self, track_id):
        """
        Genera un color único y consistente para cada ID
        """
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = (
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255))
            )
        return self.colors[track_id]
    
    def filter_players_in_field(self, detections, frame_shape):
        """
        Filtra detecciones para mantener solo jugadores relevantes
        """
        filtered_detections = []
        height, width = frame_shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            
            # Calcular centro y dimensiones
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Filtros:
            # 1. Tamaño apropiado
            min_size = min(width, height) * 0.015
            max_size = min(width, height) * 0.35
            
            if not (min_size < bbox_height < max_size):
                continue
            
            # 2. Proporción humana
            aspect_ratio = bbox_height / bbox_width
            if not (1.2 < aspect_ratio < 5.0):
                continue
            
            # 3. Posición en el campo
            if center_y < height * 0.05:
                continue
            
            filtered_detections.append(detection)
        
        return filtered_detections
    
    def detect_players_in_frame(self, frame):
        """
        Detecta jugadores en un frame con tracking usando YOLOv11
        """
        # Detección de personas con YOLOv11
        results = self.detection_model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Extraer detecciones
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    if cls == 0:  # Clase 0 = persona
                        detections.append(np.append(box, conf))
        
        # Filtrar jugadores
        filtered_detections = self.filter_players_in_field(detections, frame.shape)
        
        # Actualizar tracker
        tracked_players = self.tracker.update(filtered_detections)
        
        # Guardar IDs detectados
        for track_id, _, _, _, _, _ in tracked_players:
            self.detected_player_ids.add(track_id)
        
        # Dibujar detecciones
        frame_with_detections = self.draw_tracked_players(frame, tracked_players)
        
        return frame_with_detections, len(tracked_players), tracked_players
    
    def draw_tracked_players(self, frame, tracked_players):
        """
        Dibuja jugadores con IDs consistentes
        """
        frame_with_detections = frame.copy()
        
        for track_id, x1, y1, x2, y2, conf in tracked_players:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Color único para cada ID
            color = self.generate_color_for_id(track_id)
            
            # Dibujar caja
            cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta
            label = f'Jugador ID-{track_id}: {conf:.2f}'
            label_color = (255, 255, 255)
            
            # Fondo para la etiqueta
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_with_detections, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 5, y1), 
                         color, -1)
            
            # Texto de la etiqueta
            cv2.putText(frame_with_detections, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            
            # Punto central
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame_with_detections, (center_x, center_y), 3, color, -1)
        
        return frame_with_detections
    
    def get_pose_for_player(self, frame, bbox):
        """
        Obtiene los 17 keypoints de pose para un jugador específico usando YOLOv11-pose
        
        Args:
            frame: Frame de video
            bbox: Bounding box del jugador (x1, y1, x2, y2)
        
        Returns:
            Lista de 17 keypoints [(x1, y1, conf1), (x2, y2, conf2), ...] o None si no detecta
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Expandir ligeramente la región de interés
        height, width = frame.shape[:2]
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)
        
        # Extraer región del jugador
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return None
        
        # Ejecutar detección de pose con YOLOv11
        pose_results = self.pose_model(player_region, conf=0.3, verbose=False)
        
        for result in pose_results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # Obtener los keypoints del primer (y más confiable) resultado
                keypoints = result.keypoints.data[0].cpu().numpy()  # Shape: (17, 3) [x, y, conf]
                
                # Ajustar coordenadas al frame completo
                adjusted_keypoints = []
                for kp in keypoints:
                    if len(kp) >= 3:
                        kp_x, kp_y, kp_conf = kp[0], kp[1], kp[2]
                        # Ajustar coordenadas
                        adj_x = kp_x + x1 if kp_conf > 0.3 else np.nan
                        adj_y = kp_y + y1 if kp_conf > 0.3 else np.nan
                        adj_conf = kp_conf if kp_conf > 0.3 else np.nan
                        adjusted_keypoints.append((adj_x, adj_y, adj_conf))
                    else:
                        adjusted_keypoints.append((np.nan, np.nan, np.nan))
                
                return adjusted_keypoints
        
        # Si no se detectó pose, retornar NaN para todos los keypoints
        return [(np.nan, np.nan, np.nan)] * 17
    
    def process_video_first_pass(self, video_path, output_path=None, show_video=True):
        """
        Primera pasada: detecta y trackea jugadores usando YOLOv11
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = 1920
        # height = 1080
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Procesando video: {video_path}")
        print(f"📏 Resolución: {width}x{height}, FPS: {fps}, Frames totales: {total_frames}")
        print("🔍 PRIMERA PASADA: Detectando y trackeando jugadores con YOLOv11...")
        
        # Writer para video de salida
        writer = None
        if output_path:
            # Intentar H.264 primero (mejor compatibilidad con navegadores)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'H264')
            except:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        self.player_counts = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # frame = cv2.resize(frame, (1568, 1045))
                
                # Detectar y trackear jugadores
                frame_with_detections, player_count, tracked_players = self.detect_players_in_frame(frame)
                self.player_counts.append(player_count)
                
                # Información del frame
                info_text = f"Frame: {frame_count+1}/{total_frames} | Jugadores: {player_count} | YOLOv11"
                cv2.putText(frame_with_detections, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Lista de IDs detectados
                if tracked_players:
                    ids_text = f"IDs activos: {', '.join([str(tid) for tid, _, _, _, _, _ in tracked_players])}"
                    cv2.putText(frame_with_detections, ids_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Mostrar video
                if show_video:
                    cv2.imshow('Primera Pasada: Detección YOLOv11 - Presiona Q para salir', frame_with_detections)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Guardar frame
                if writer:
                    writer.write(frame_with_detections)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progreso: {progress:.1f}%")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_video:
                cv2.destroyAllWindows()
        
        return self.detected_player_ids
    
    def select_players_interactive(self, detected_ids):
        """
        Permite al usuario seleccionar uno o múltiples jugadores ID
        """
        print("\n" + "="*60)
        print("🎯 SELECCIÓN DE JUGADORES PARA ANÁLISIS DE POSE")
        print("="*60)
        print("Durante el análisis se detectaron los siguientes jugadores:")
        print("-" * 40)
        
        sorted_ids = sorted(detected_ids)
        for i, player_id in enumerate(sorted_ids, 1):
            print(f"  {i}. Jugador ID-{player_id}")
        
        print("-" * 40)
        print("💡 OPCIONES DE SELECCIÓN:")
        print("  • Un jugador: Ingresa el número (ej: 2)")
        print("  • Múltiples jugadores: Separa por comas (ej: 1,3,5)")
        print("  • Todos los jugadores: Ingresa 'todos' o 'all'")
        print("  • Cancelar: Ingresa 'q' o 'quit'")
        print("-" * 40)
        
        while True:
            try:
                selection = input(f"\n¿Qué jugadores quieres analizar? (1-{len(sorted_ids)}): ").strip()
                
                if selection.lower() in ['q', 'quit', 'salir']:
                    print("❌ Análisis cancelado por el usuario.")
                    return None
                
                if selection.lower() in ['todos', 'all', 'todo']:
                    selected_ids = sorted_ids.copy()
                    print(f"✅ Has seleccionado TODOS los jugadores: {selected_ids}")
                else:
                    # Procesar selección múltiple
                    selections = [s.strip() for s in selection.split(',')]
                    selected_ids = []
                    
                    for sel in selections:
                        try:
                            sel_idx = int(sel) - 1
                            if 0 <= sel_idx < len(sorted_ids):
                                selected_ids.append(sorted_ids[sel_idx])
                            else:
                                print(f"❌ Número inválido: {sel} (debe estar entre 1 y {len(sorted_ids)})")
                                selected_ids = []
                                break
                        except ValueError:
                            print(f"❌ '{sel}' no es un número válido")
                            selected_ids = []
                            break
                    
                    if not selected_ids:
                        print("Intenta nuevamente.")
                        continue
                    
                    # Eliminar duplicados y ordenar
                    selected_ids = sorted(list(set(selected_ids)))
                    print(f"✅ Has seleccionado: Jugadores {selected_ids}")
                
                return selected_ids
                    
            except KeyboardInterrupt:
                print("\n❌ Análisis cancelado por el usuario.")
                return None
    
    def process_video_second_pass(self, video_path, selected_player_ids, csv_output_path):
        """
        Segunda pasada: extrae landmarks de los jugadores seleccionados usando YOLOv11-pose
        Combina múltiples IDs eligiendo el mejor por frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Reiniciar tracker para segunda pasada
        self.tracker = PlayerTracker(max_distance=80, max_frames_lost=15)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n🎯 SEGUNDA PASADA: Extrayendo landmarks con YOLOv11-pose de {len(selected_player_ids)} jugadores")
        print(f"👥 Jugadores candidatos: {selected_player_ids}")
        if len(selected_player_ids) > 1:
            print("🧠 Estrategia: En cada frame se elige el jugador con más landmarks válidos")
        print(f"💾 Los datos se guardarán en: {csv_output_path}")
        print("🦴 Extrayendo 17 keypoints corporales por frame...")
        
        # Preparar datos para CSV
        csv_data = []
        frame_count = 0
        
        # Estadísticas por jugador
        player_usage_count = {pid: 0 for pid in selected_player_ids}
        total_landmarks_detected = 0
        
        # Crear columnas para CSV (estructura simple)
        columns = ['frame']
        for kp_name in self.keypoint_names:
            columns.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_confidence'])
        
        print(f"📊 Estructura CSV: {len(columns)} columnas (frame + 17 keypoints × 3 valores)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detectar y trackear jugadores con YOLOv11
                results = self.detection_model(frame, conf=self.confidence_threshold, verbose=False)
                
                detections = []
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confidences, classes):
                            if cls == 0:
                                detections.append(np.append(box, conf))
                
                filtered_detections = self.filter_players_in_field(detections, frame.shape)
                tracked_players = self.tracker.update(filtered_detections)
                
                # Encontrar jugadores candidatos este frame
                candidate_players = {}
                for track_id, x1, y1, x2, y2, conf in tracked_players:
                    if track_id in selected_player_ids:
                        # Extraer landmarks para este jugador candidato
                        bbox = (x1, y1, x2, y2)
                        keypoints = self.get_pose_for_player(frame, bbox)
                        
                        # Contar landmarks válidos (no NaN)
                        valid_landmarks_count = 0
                        for kp in keypoints:
                            if not (pd.isna(kp[0]) or pd.isna(kp[1]) or pd.isna(kp[2])):
                                valid_landmarks_count += 1
                        
                        candidate_players[track_id] = {
                            'keypoints': keypoints,
                            'valid_count': valid_landmarks_count
                        }
                
                # Elegir el mejor jugador para este frame
                best_player_id = None
                best_keypoints = None
                
                if candidate_players:
                    # Encontrar el jugador con más landmarks válidos
                    best_player_id = max(candidate_players.keys(), 
                                       key=lambda pid: candidate_players[pid]['valid_count'])
                    best_keypoints = candidate_players[best_player_id]['keypoints']
                    
                    # Actualizar estadísticas
                    player_usage_count[best_player_id] += 1
                    if candidate_players[best_player_id]['valid_count'] > 0:
                        total_landmarks_detected += 1
                
                # Preparar fila de datos
                if best_player_id is not None and best_keypoints is not None:
                    # Usar el mejor jugador encontrado
                    row_data = [frame_count]
                    for kp in best_keypoints:
                        row_data.extend([kp[0], kp[1], kp[2]])
                else:
                    # No se encontró ningún jugador candidato - usar NaN
                    row_data = [frame_count]
                    for _ in range(17):  # 17 keypoints
                        row_data.extend([np.nan, np.nan, np.nan])
                
                csv_data.append(row_data)
                frame_count += 1
                
                # Mostrar progreso cada 50 frames
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    
                    if len(selected_player_ids) > 1:
                        # Mostrar distribución de uso para múltiples jugadores
                        usage_info = []
                        for pid in selected_player_ids:
                            usage_pct = (player_usage_count[pid] / frame_count) * 100
                            usage_info.append(f"ID-{pid}: {usage_pct:.1f}%")
                        
                        landmarks_rate = (total_landmarks_detected / frame_count) * 100
                        print(f"Progreso: {progress:.1f}% | Uso: {', '.join(usage_info)} | Landmarks: {landmarks_rate:.1f}%")
                    else:
                        # Para un solo jugador, mostrar detección simple
                        landmarks_rate = (total_landmarks_detected / frame_count) * 100
                        print(f"Progreso: {progress:.1f}% | Landmarks detectados: {landmarks_rate:.1f}%")
        
        finally:
            cap.release()
        
        # Guardar datos en CSV
        df = pd.DataFrame(csv_data, columns=columns)
        df.to_csv(csv_output_path, index=False)
        
        print(f"\n✅ Análisis completado!")
        print(f"📊 Frames procesados: {frame_count}")
        print(f"🦴 Frames con landmarks: {total_landmarks_detected} ({(total_landmarks_detected/frame_count)*100:.1f}%)")
        
        if len(selected_player_ids) > 1:
            print(f"👥 Distribución de uso por jugador:")
            for player_id in selected_player_ids:
                usage_count = player_usage_count[player_id]
                usage_rate = (usage_count / frame_count) * 100 if frame_count > 0 else 0
                print(f"   🎯 Jugador ID-{player_id}: {usage_count} frames ({usage_rate:.1f}%)")
        
        print(f"💾 Datos guardados en: {csv_output_path}")
        
        return player_usage_count, frame_count
    
    def calculate_statistics(self):
        """
        Calcula estadísticas básicas de detección
        """
        if not self.player_counts:
            return {}
        
        stats = {
            'total_frames': len(self.player_counts),
            'jugadores_promedio': np.mean(self.player_counts),
            'jugadores_max': np.max(self.player_counts),
            'jugadores_min': np.min(self.player_counts),
            'jugadores_mediana': np.median(self.player_counts),
            'tracks_unicos': len(self.detected_player_ids)
        }
        
        return stats
    
    def print_statistics(self, stats):
        """
        Imprime estadísticas de detección
        """
        print("\n" + "="*60)
        print("📈 ESTADÍSTICAS DE DETECCIÓN")
        print("="*60)
        print(f"Total de frames procesados: {stats['total_frames']}")
        print(f"Jugadores promedio por frame: {stats['jugadores_promedio']:.1f}")
        print(f"Máximo jugadores detectados: {stats['jugadores_max']}")
        print(f"Mínimo jugadores detectados: {stats['jugadores_min']}")
        print(f"Mediana de jugadores: {stats['jugadores_mediana']:.1f}")
        print(f"IDs únicos detectados: {stats['tracks_unicos']}")

def main():
    parser = argparse.ArgumentParser(description='Detector de jugadores con análisis de landmarks usando YOLOv11')
    parser.add_argument('video_path', help='Ruta del video de entrada')
    parser.add_argument('--output-video', '-ov', help='Ruta del video de salida con tracking (opcional)')
    parser.add_argument('--output-csv', '-oc', help='Nombre del archivo CSV para landmarks (opcional)')
    parser.add_argument('--model', '-m', help='Ruta del modelo YOLOv11 personalizado (opcional)')
    parser.add_argument('--confidence', '-c', type=float, default=0.4, 
                       help='Umbral de confianza (default: 0.4)')
    parser.add_argument('--no-display', action='store_true', 
                       help='No mostrar video en tiempo real')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: El archivo {args.video_path} no existe")
        return
    
    # Generar nombre de archivo CSV basado en el video
    if args.output_csv:
        csv_output = args.output_csv
    else:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_output = f"./csvs/{video_name}.csv"
    
    # Crear directorio csvs si no existe
    os.makedirs(os.path.dirname(csv_output) if os.path.dirname(csv_output) else '.', exist_ok=True)
    
    # Crear detector
    detector = FootballPlayerDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    print("🚀 INICIANDO ANÁLISIS DE LANDMARKS DE JUGADORES CON YOLOv11")
    print("=" * 60)
    print("✅ Tracking de IDs consistente")
    print("✅ Detección de 17 keypoints corporales con YOLOv11-pose") 
    print("✅ Selección interactiva de jugadores (uno o múltiples)")
    print("✅ Exportación a CSV")
    print("✅ Nombres de archivo basados en el video")
    print("✅ Modelos YOLOv11 optimizados")
    
    try:
        # PRIMERA PASADA: Detección y tracking
        detected_ids = detector.process_video_first_pass(
            video_path=args.video_path,
            output_path=args.output_video,
            show_video=not args.no_display
        )
        
        if not detected_ids:
            print("❌ No se detectaron jugadores en el video.")
            return
        
        # Mostrar estadísticas
        stats = detector.calculate_statistics()
        detector.print_statistics(stats)
        
        # SELECCIÓN INTERACTIVA (múltiples jugadores)
        selected_ids = detector.select_players_interactive(detected_ids)
        
        if selected_ids is None:
            print("❌ No se seleccionó ningún jugador. Terminando análisis.")
            return
        
        # SEGUNDA PASADA: Extracción de landmarks
        player_usage_stats, total_frames = detector.process_video_second_pass(
            video_path=args.video_path,
            selected_player_ids=selected_ids,
            csv_output_path=csv_output
        )
        
        # Resumen final
        print("\n" + "="*60)
        print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        if len(selected_ids) == 1:
            print(f"🎯 Jugador analizado: ID-{selected_ids[0]}")
        else:
            print(f"👥 Jugadores candidatos: {selected_ids}")
            print("🧠 Estrategia: Se eligió automáticamente el mejor jugador por frame")
        
        print(f"🎹 Frames totales: {total_frames}")
        
        if len(selected_ids) > 1:
            # Mostrar distribución final para múltiples jugadores
            print("📊 Distribución de uso final:")
            for player_id in selected_ids:
                usage_count = player_usage_stats.get(player_id, 0)
                usage_rate = (usage_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"   🎯 Jugador ID-{player_id}: {usage_count} frames ({usage_rate:.1f}%)")
        
        print(f"💾 Archivo CSV: {csv_output}")
        print(f"🤖 Procesado con YOLOv11 + YOLOv11-pose")
        
        if args.output_video:
            print(f"🎬 Video con tracking: {args.output_video}")
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Ejemplo de uso programático con YOLOv11
def ejemplo_completo_yolov11():
    """
    Ejemplo de uso completo del sistema con YOLOv11 y selección múltiple
    """
    video_path = "penal_futbol.mp4"
    
    detector = FootballPlayerDetector(confidence_threshold=0.4)
    
    try:
        print("🚀 Ejecutando análisis completo con YOLOv11...")
        
        # Primera pasada
        detected_ids = detector.process_video_first_pass(
            video_path=video_path,
            output_path="video_con_tracking_yolov11.mp4",
            show_video=True
        )
        
        if detected_ids:
            # Mostrar estadísticas
            stats = detector.calculate_statistics()
            detector.print_statistics(stats)
            
            # Selección interactiva (múltiples jugadores)
            selected_ids = detector.select_players_interactive(detected_ids)
            
            if selected_ids:
                # Generar nombre de CSV basado en el video
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                csv_filename = f"{video_name}_yolov11.csv"
                
                # Segunda pasada
                player_usage_stats, total_frames = detector.process_video_second_pass(
                    video_path=video_path,
                    selected_player_ids=selected_ids,
                    csv_output_path=csv_filename
                )
                
                if len(selected_ids) == 1:
                    print(f"✅ ¡Análisis completo! Landmarks guardados para Jugador ID-{selected_ids[0]}")
                else:
                    print(f"✅ ¡Análisis completo! Landmarks combinados de jugadores: {selected_ids}")
                
                print(f"📄 Archivo generado: {csv_filename}")
                print(f"🤖 Procesado con YOLOv11")
        else:
            print("❌ No se detectaron jugadores.")
            
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {video_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Descomenta la siguiente línea para ejecutar el ejemplo
# ejemplo_completo_yolov11()