"""
FastAPI Backend principal pentru bot-ul anti-toxicitate
REST API pentru comunicarea cu Discord Bot și React Dashboard
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from contextlib import asynccontextmanager

# FastAPI și dependențe
from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Pydantic pentru validare
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from bson.errors import InvalidId

# JWT pentru autentificare
import jwt
from passlib.context import CryptContext

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Imports locale
from config import get_config
from database import get_database, ServerModel, UserModel, MessageModel
from ai_detector import get_ai_detector, analyze_toxicity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurare globală
config = get_config()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# === PYDANTIC MODELS PENTRU API ===

class PyObjectId(ObjectId):
    """Custom ObjectId pentru Pydantic"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid ObjectId')
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='string')

class BaseResponse(BaseModel):
    """Model de bază pentru răspunsuri"""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseResponse):
    """Model pentru răspunsuri de eroare"""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class AnalyzeMessageRequest(BaseModel):
    """Request pentru analiza mesajelor"""
    message: str = Field(..., min_length=1, max_length=2000)
    server_id: Optional[int] = None
    user_id: Optional[int] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Mesajul nu poate fi gol')
        return v.strip()

class AnalyzeMessageResponse(BaseResponse):
    """Response pentru analiza mesajelor"""
    analysis: Dict[str, Any]
    recommendations: List[str] = Field(default_factory=list)

class ServerConfigRequest(BaseModel):
    """Request pentru configurarea serverului"""
    toxicity_threshold: float = Field(50.0, ge=0, le=100)
    auto_moderate: bool = True
    feedback_channel: Optional[int] = None
    language: str = Field('ro', regex='^(ro|en|auto)$')
    strict_mode: bool = False
    auto_delete_toxic: bool = False
    
class ServerStatsResponse(BaseResponse):
    """Response pentru statisticile serverului"""
    server_id: int
    stats: Dict[str, Any]
    daily_stats: List[Dict[str, Any]]
    top_categories: Dict[str, int]

class UserStatsResponse(BaseResponse):
    """Response pentru statisticile utilizatorului"""
    user_id: int
    username: str
    stats: Dict[str, Any]
    recent_toxic_messages: List[Dict[str, Any]]

class ModerationActionRequest(BaseModel):
    """Request pentru acțiuni de moderare"""
    user_id: int
    action_type: str = Field(..., regex='^(warning|timeout|ban|kick)$')
    duration: Optional[int] = Field(None, ge=0)
    reason: Optional[str] = None

class TopUsersResponse(BaseResponse):
    """Response pentru top utilizatori toxici"""
    users: List[Dict[str, Any]]
    total_count: int

# === DEPENDENCY INJECTION ===

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Obține utilizatorul curent din JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, config.fastapi.secret_key, algorithms=[config.fastapi.algorithm])
        user_id: int = payload.get("user_id")
        username: str = payload.get("username")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalid",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {"user_id": user_id, "username": username}
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalid sau expirat",
            headers={"WWW-Authenticate": "Bearer"},
        )

def validate_server_access(server_id: int, current_user: dict = Depends(get_current_user)):
    """Validează accesul utilizatorului la server"""
    # În implementarea reală, aici ai verifica dacă utilizatorul
    # are permisiuni de admin pe serverul Discord respectiv
    # Pentru moment, returnăm True
    return True

# === LIFECYCLE MANAGEMENT ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionează inițializarea și curățarea aplicației"""
    
    # Startup
    logger.info("🚀 Inițializare FastAPI backend...")
    
    try:
        # Inițializează database
        db = await get_database()
        logger.info("✅ Database conectat")
        
        # Inițializează AI detector
        ai_detector = await get_ai_detector()
        logger.info("✅ AI Detector inițializat")
        
        # Creează task-uri în background
        asyncio.create_task(periodic_stats_update())
        logger.info("✅ Background tasks pornite")
        
        logger.info("🎉 FastAPI backend gata!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Eroare la inițializare: {e}")
        raise
    
    # Shutdown
    logger.info("🔄 Oprire FastAPI backend...")
    
    try:
        # Curăță resursele
        db = await get_database()
        await db.close()
        logger.info("✅ Database deconectat")
        
    except Exception as e:
        logger.error(f"❌ Eroare la oprire: {e}")
    
    logger.info("👋 FastAPI backend oprit")

# === APLICAȚIA FASTAPI ===

app = FastAPI(
    title="Discord Anti-Toxicity Bot API",
    description="REST API pentru bot-ul de detectare și moderare a toxicității",
    version=config.version,
    docs_url="/docs" if config.fastapi.debug else None,
    redoc_url="/redoc" if config.fastapi.debug else None,
    lifespan=lifespan
)

# === MIDDLEWARE ===

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.fastapi.allow_origins,
    allow_credentials=config.fastapi.allow_credentials,
    allow_methods=config.fastapi.allow_methods,
    allow_headers=config.fastapi.allow_headers,
)

# Trusted hosts pentru securitate
if config.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# === EXCEPTION HANDLERS ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler pentru excepțiile HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            details={"path": str(request.url.path)}
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler pentru excepțiile generale"""
    logger.error(f"Eroare neașteptată: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Eroare internă a serverului",
            error_code="INTERNAL_ERROR",
            details={"path": str(request.url.path)} if config.fastapi.debug else None
        ).dict()
    )

# === ENDPOINT-URI PRINCIPALE ===

@app.get("/", response_model=BaseResponse)
async def root():
    """Endpoint de bază pentru verificarea statusului"""
    return BaseResponse(
        message=f"Discord Anti-Toxicity Bot API v{config.version}"
    )

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check complet pentru toate componentele"""
    
    health_status = {
        "api": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": config.version,
        "environment": config.environment
    }
    
    try:
        # Verifică database
        db = await get_database()
        db_health = await db.get_health_check()
        health_status["database"] = db_health["status"]
        health_status["collections"] = db_health.get("collections", {})
        
    except Exception as e:
        health_status["database"] = "unhealthy"
        health_status["database_error"] = str(e)
    
    try:
        # Verifică AI detector
        ai_detector = await get_ai_detector()
        ai_stats = ai_detector.get_stats()
        health_status["ai_detector"] = "healthy" if ai_stats["model_loaded"] else "unhealthy"
        health_status["ai_stats"] = ai_stats
        
    except Exception as e:
        health_status["ai_detector"] = "unhealthy"
        health_status["ai_error"] = str(e)
    
    # Determină statusul general
    if all(status != "unhealthy" for key, status in health_status.items() 
           if key.endswith(("database", "ai_detector"))):
        health_status["overall"] = "healthy"
        status_code = 200
    else:
        health_status["overall"] = "degraded"
        status_code = 503
    
    return JSONResponse(content=health_status, status_code=status_code)

# === ENDPOINT-URI ANALIZĂ TOXICITATE ===

@app.post("/api/analyze", response_model=AnalyzeMessageResponse)
@limiter.limit("30/minute")
async def analyze_message(
    request: Request,
    data: AnalyzeMessageRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Analizează un mesaj pentru toxicitate"""
    
    try:
        # Analizează mesajul cu AI
        analysis_result = await analyze_toxicity(data.message, data.server_id)
        
        # Generează recomandări
        recommendations = []
        if analysis_result.is_toxic:
            if "profanity" in analysis_result.categories:
                recommendations.append("Evită limbajul vulgar pentru o comunicare mai eficientă")
            if "harassment" in analysis_result.categories:
                recommendations.append("Încearcă să îți exprimi punctul de vedere fără a ataca persoana")
            if "hate_speech" in analysis_result.categories:
                recommendations.append("Limbajul discriminatoriu nu este tolerat")
        else:
            recommendations.append("Mesajul pare să aibă un ton constructiv!")
        
        # Salvează în background dacă avem server_id și user_id
        if data.server_id and data.user_id:
            background_tasks.add_task(
                save_analysis_to_db,
                data.server_id,
                data.user_id,
                data.message,
                analysis_result.dict()
            )
        
        return AnalyzeMessageResponse(
            analysis=analysis_result.dict(),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Eroare la analiza mesajului: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eroare la procesarea mesajului"
        )

@app.get("/api/servers/{server_id}/stats", response_model=ServerStatsResponse)
@limiter.limit("10/minute")
async def get_server_stats(
    request: Request,
    server_id: int,
    days: int = 30,
    current_user: dict = Depends(get_current_user),
    server_access = Depends(validate_server_access)
):
    """Obține statisticile pentru un server"""
    
    try:
        db = await get_database()
        
        # Obține serverul
        server = await db.get_server_by_discord_id(server_id)
        if not server:
            raise HTTPException(
                status_code=404,
                detail="Serverul nu a fost găsit"
            )
        
        # Obține analytics
        analytics = await db.get_server_analytics(server.id, days)
        
        return ServerStatsResponse(
            server_id=server_id,
            stats=analytics["summary"],
            daily_stats=analytics["daily_stats"],
            top_categories=analytics["top_categories"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la obținerea statisticilor server: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eroare la procesarea cererii"
        )

@app.get("/api/servers/{server_id}/users/top", response_model=TopUsersResponse)
@limiter.limit("5/minute")
async def get_top_toxic_users(
    request: Request,
    server_id: int,
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    server_access = Depends(validate_server_access)
):
    """Obține top utilizatori cu toxicitate ridicată"""
    
    try:
        db = await get_database()
        
        # Obține serverul
        server = await db.get_server_by_discord_id(server_id)
        if not server:
            raise HTTPException(
                status_code=404,
                detail="Serverul nu a fost găsit"
            )
        
        # Obține top utilizatori
        top_users = await db.get_top_toxic_users(server.id, limit)
        
        return TopUsersResponse(
            users=top_users,
            total_count=len(top_users)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la obținerea top utilizatori: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eroare la procesarea cererii"
        )

@app.get("/api/users/{user_id}/stats", response_model=UserStatsResponse)
@limiter.limit("20/minute")
async def get_user_stats(
    request: Request,
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Obține statisticile pentru un utilizator"""
    
    try:
        db = await get_database()
        
        # Obține utilizatorul
        user = await db.get_user_by_discord_id(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail="Utilizatorul nu a fost găsit"
            )
        
        # Obține mesajele toxice recente
        recent_toxic = await db.get_user_toxic_messages(user.id, limit=10)
        
        # Formatează mesajele pentru response
        formatted_messages = []
        for msg in recent_toxic:
            formatted_messages.append({
                "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                "toxicity_score": msg.analysis.toxicity_score,
                "categories": msg.analysis.categories,
                "created_at": msg.created_at.isoformat(),
                "bypass_detected": msg.analysis.bypass_detected
            })
        
        return UserStatsResponse(
            user_id=user_id,
            username=user.username,
            stats={
                "total_messages": user.total_messages,
                "toxic_messages": user.toxic_messages,
                "warnings": user.warnings,
                "toxicity_ratio": user.toxicity_ratio,
                "last_activity": user.last_activity.isoformat() if user.last_activity else None
            },
            recent_toxic_messages=formatted_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la obținerea statisticilor utilizator: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eroare la procesarea cererii"
        )

@app.post("/api/servers/{server_id}/config", response_model=BaseResponse)
@limiter.limit("5/minute")
async def update_server_config(
    request: Request,
    server_id: int,
    config_data: ServerConfigRequest,
    current_user: dict = Depends(get_current_user),
    server_access = Depends(validate_server_access)
):
    """Actualizează configurația unui server"""
    
    try:
        db = await get_database()
        
        # Obține sau creează serverul
        server = await db.get_server_by_discord_id(server_id)
        if not server:
            # Creează server nou cu configurația default
            server = await db.create_or_update_server(
                discord_id=server_id,
                name=f"Server {server_id}",
                settings=config_data.dict()
            )
        else:
            # Actualizează configurația existentă
            await db.update_server_settings(server_id, config_data.dict())
        
        return BaseResponse(
            message="Configurația serverului a fost actualizată cu succes"
        )
        
    except Exception as e:
        logger.error(f"Eroare la actualizarea configurației server: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eroare la actualizarea configurației"
        )

@app.post("/api/moderation/action", response_model=BaseResponse)
@limiter.limit("10/minute")
async def create_moderation_action(
    request: Request,
    action_data: ModerationActionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Creează o acțiune de moderare"""
    
    try:
        db = await get_database()
        
        # Obține utilizatorul target
        user = await db.get_user_by_discord_id(action_data.user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail="Utilizatorul nu a fost găsit"
            )
        
        # Validează durata pentru timeout/ban
        if action_data.action_type in ['timeout', 'ban'] and not action_data.duration:
            raise HTTPException(
                status_code=400,
                detail="Durata este obligatorie pentru timeout și ban"
            )
        
        # Pentru moment, salvăm doar în baza de date
        # În implementarea completă, aici ai trimite comanda la bot-ul Discord
        # prin webhook sau mesaj în coadă
        
        return BaseResponse(
            message=f"Acțiunea {action_data.action_type} a fost înregistrată pentru utilizatorul {action_data.user_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la crearea acțiunii de moderare: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eroare la procesarea acțiunii"
        )

# === BACKGROUND TASKS ===

async def save_analysis_to_db(server_id: int, user_id: int, content: str, analysis: Dict):
    """Salvează analiza în baza de date (background task)"""
    try:
        db = await get_database()
        
        # Obține/creează serverul și utilizatorul
        server = await db.get_server_by_discord_id(server_id)
        if not server:
            server = await db.create_or_update_server(
                discord_id=server_id,
                name=f"Server {server_id}"
            )
        
        user = await db.get_user_by_discord_id(user_id)
        if not user:
            user = await db.create_or_update_user(
                discord_id=user_id,
                username=f"User {user_id}"
            )
        
        # Salvează mesajul (folosim timestamp ca message_id fictiv)
        fake_message_id = int(datetime.utcnow().timestamp() * 1000000)
        
        await db.save_message_analysis(
            discord_message_id=fake_message_id,
            server_id=server.id,
            user_id=user.id,
            channel_id=0,  # Canal necunoscut pentru analiză manuală
            content=content,
            analysis_result=analysis
        )
        
        # Actualizează contoarele utilizatorului
        await db.increment_user_message_count(user_id, analysis.get('is_toxic', False))
        
        logger.info(f"Analiză salvată: server={server_id}, user={user_id}, toxic={analysis.get('is_toxic', False)}")
        
    except Exception as e:
        logger.error(f"Eroare la salvarea analizei în DB: {e}")

async def periodic_stats_update():
    """Task periodic pentru actualizarea statisticilor"""
    while True:
        try:
            await asyncio.sleep(3600)  # Rulează la fiecare oră
            
            db = await get_database()
            
            # Obține toate serverele active
            servers_collection = db.get_collection('servers')
            
            async for server_doc in servers_collection.find({"is_active": True}):
                try:
                    # Actualizează statisticile zilnice pentru fiecare server
                    await db.update_daily_stats(server_doc["_id"])
                    logger.debug(f"Stats actualizate pentru server {server_doc['discord_id']}")
                    
                except Exception as e:
                    logger.error(f"Eroare la actualizarea stats pentru server {server_doc['discord_id']}: {e}")
            
            logger.info("📊 Task periodic stats completat")
            
        except Exception as e:
            logger.error(f"Eroare în task periodic stats: {e}")

# === MIDDLEWARE PENTRU LOGGING ===

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pentru logging requests"""
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    if config.fastapi.log_requests:
        process_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
    
    return response

# === WEBHOOK ENDPOINT PENTRU DISCORD BOT ===

@app.post("/api/webhook/discord/message")
@limiter.limit("100/minute")
async def discord_message_webhook(
    request: Request,
    message_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Webhook pentru primirea mesajelor de la bot-ul Discord"""
    
    # Verifică autentificarea webhook-ului
    auth_header = request.headers.get("Authorization")
    expected_token = f"Bearer {config.fastapi.secret_key}"
    
    if auth_header != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Webhook unauthorized"
        )
    
    try:
        # Procesează mesajul în background
        background_tasks.add_task(process_discord_message, message_data)
        
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"Eroare la procesarea webhook Discord: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eroare la procesarea mesajului"
        )

async def process_discord_message(message_data: Dict[str, Any]):
    """Procesează un mesaj venit de la Discord bot"""
    try:
        # Extrage datele mesajului
        content = message_data.get("content", "")
        server_id = message_data.get("server_id")
        user_id = message_data.get("user_id")
        channel_id = message_data.get("channel_id")
        message_id = message_data.get("message_id")
        
        if not all([content, server_id, user_id, channel_id, message_id]):
            logger.warning("Date incomplete în webhook Discord")
            return
        
        # Analizează mesajul
        analysis_result = await analyze_toxicity(content, server_id)
        
        # Salvează în baza de date
        db = await get_database()
        
        # Obține/creează entitățile necesare
        server = await db.get_server_by_discord_id(server_id)
        if not server:
            server = await db.create_or_update_server(
                discord_id=server_id,
                name=f"Server {server_id}"
            )
        
        user = await db.get_user_by_discord_id(user_id)
        if not user:
            user = await db.create_or_update_user(
                discord_id=user_id,
                username=f"User {user_id}"
            )
        
        # Salvează mesajul
        await db.save_message_analysis(
            discord_message_id=message_id,
            server_id=server.id,
            user_id=user.id,
            channel_id=channel_id,
            content=content,
            analysis_result=analysis_result.dict()
        )
        
        # Actualizează contoarele
        await db.increment_user_message_count(user_id, analysis_result.is_toxic)
        
        logger.info(f"Mesaj Discord procesat: {message_id}, toxic={analysis_result.is_toxic}")
        
    except Exception as e:
        logger.error(f"Eroare la procesarea mesajului Discord: {e}")

# === FUNCȚIA PRINCIPALĂ ===

def create_app() -> FastAPI:
    """Factory pentru crearea aplicației FastAPI"""
    return app

if __name__ == "__main__":
    # Rulează serverul în modul development
    uvicorn.run(
        "main:app",
        host=config.fastapi.host,
        port=config.fastapi.port,
        reload=config.fastapi.reload,
        log_level="info"
    )