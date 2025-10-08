# PENAL - Sistema Inteligente de Análisis de Penales ⚽

Sistema web para análisis de penales mediante Machine Learning, con almacenamiento en AWS S3, base de datos PostgreSQL en RDS y predicciones con SageMaker.

## 🏗️ Arquitectura

```
Frontend (React) ←→ Backend (Flask/Python) ←→ AWS Services
                                              ├─ RDS PostgreSQL
                                              ├─ S3 (videos)
                                              └─ SageMaker (predicciones)
```

## 📋 Requisitos Previos

- Node.js 16+ y npm
- Python 3.8+
- PostgreSQL (o RDS configurado)
- Cuenta AWS (para S3, RDS, SageMaker)

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd proyecto-penal
```

### 2. Configurar Backend

```bash
cd backend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### 3. Configurar Frontend

```bash
cd frontend

# Instalar dependencias
npm install

# (Opcional) Configurar variables de entorno si es necesario
```


## 🎮 Uso

### Iniciar Backend

```bash
cd backend
python app.py
```

El servidor Flask correrá en `http://localhost:5000`

### Iniciar Frontend

```bash
cd frontend
npm start
```

La aplicación React correrá en `http://localhost:3000`

## 📁 Estructura del Proyecto

```
proyecto-penal/
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx              # Componente principal
│   │   ├── index.js             # Entry point
│   │   └── index.css            # Estilos globales
│   ├── package.json
│   └── tailwind.config.js
│
├── backend/
│   ├── app.py                   # API Flask
│   ├── config.py                # Configuración
│   ├── .env                     # Variables de entorno (no subir a git)
│   └── requirements.txt         # Dependencias Python
│
├── .gitignore
└── README.md
```

## 🔧 Configuración de .env

Crear archivo `.env` en la carpeta `backend/` con:

```env
DB_HOST=your-rds-endpoint.rds.amazonaws.com
DB_NAME=penal_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432

AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=penal-videos-bucket
```

## 🎯 Funcionalidades

### ✅ Implementadas
- **Ver Jugadores**: Lista y búsqueda de jugadores desde PostgreSQL
- **Navegación**: Sistema de navegación entre vistas

### 🚧 En Desarrollo
- Ver Videos con Posturas
- Cargar Video Nuevo
- Predicción de Penal

## 🔌 API Endpoints

### Jugadores

- `GET /api/players` - Obtener todos los jugadores
- `GET /api/players/<id>` - Obtener un jugador específico
- `GET /api/health` - Health check de la API y DB

## 🛠️ Tecnologías

**Frontend:**
- React 18
- Tailwind CSS
- Lucide React (iconos)

**Backend:**
- Flask
- PostgreSQL (psycopg2)
- python-dotenv

**AWS:**
- RDS PostgreSQL
- S3 (próximamente)
- SageMaker (próximamente)

## 📝 Notas de Desarrollo

- El frontend usa navegación interna con estado (no react-router)
- Todas las credenciales deben estar en el archivo `.env`
- CORS está habilitado para desarrollo local
