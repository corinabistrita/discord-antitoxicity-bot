"""
Modulul de configurație centralizată pentru bot-ul anti-toxicitate
MongoDB + FastAPI + React Stack
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Încarcă variabilele de mediu
load_dotenv()

@dataclass
class MongoDBConfig:
    """Configurație MongoDB"""
    host: str = os.getenv('MONGODB_HOST', 'localhost')
    port: int = int(os.getenv('MONGODB_PORT', '27017'))
    database: str = os.getenv('MONGODB_DATABASE', 'antitoxicity_bot')
    username: str = os.getenv('MONGODB_USERNAME', '')
    password: str = os.getenv('MONGODB_PASSWORD', '')
    
    # Opțiuni conexiune
    max_pool_size: int = int(os.getenv('MONGODB_POOL_SIZE', '10'))
    min_pool_size: int = int(os.getenv('MONGODB_MIN_POOL_SIZE', '1'))
    max_idle_time: int = int(os.getenv('MONGODB_MAX_IDLE_TIME', '30000'))  # ms
    
    # SSL și autentificare
    use_ssl: bool = os.getenv('MONGODB_USE_SSL', 'false').lower() == 'true'
    auth_source: str = os.getenv('MONGODB_AUTH_SOURCE', 'admin')
    replica_set: str = os.getenv('MONGODB_REPLICA_SET', '')
    
    @property
    def connection_string(self) -> str:
        """Returnează connection string-ul MongoDB"""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        
        base_url = f"mongodb://{auth}{self.host}:{self.port}/{self.database}"
        
        # Adaugă parametri de conexiune
        params = []
        if self.auth_source and self.username:
            params.append(f"authSource={self.auth_source}")
        if self.replica_set:
            params.append(f"replicaSet={self.replica_set}")
        if self.use_ssl:
            params.append("ssl=true")
        
        if params:
            base_url += "?" + "&".join(params)
        
        return base_url
    
    @property
    def motor_options(self) -> Dict[str, Any]:
        """Opțiuni pentru motor (driver async MongoDB)"""
        return {
            'maxPoolSize': self.max_pool_size,
            'minPoolSize': self.min_pool_size,
            'maxIdleTimeMS': self.max_idle_time,
            'serverSelectionTimeoutMS': 5000,  # 5 secunde timeout
            'connectTimeoutMS': 10000,  # 10 secunde pentru conectare
            'socketTimeoutMS': 20000,   # 20 secunde pentru operațiuni
        }

@dataclass
class RedisConfig:
    """Configurație Redis pentru cache și sesiuni"""
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', '6379'))
    password: str = os.getenv('REDIS_PASSWORD', '')
    db: int = int(os.getenv('REDIS_DB', '0'))
    
    # Pool settings
    max_connections: int = int(os.getenv('REDIS_MAX_CONNECTIONS', '20'))
    
    # Cache TTL (Time To Live)
    default_ttl: int = int(os.getenv('REDIS_DEFAULT_TTL', '3600'))  # 1 oră
    session_ttl: int = int(os.getenv('REDIS_SESSION_TTL', '86400'))  # 24 ore
    
    @property
    def url(self) -> str:
        """URL Redis pentru conexiune"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class FastAPIConfig:
    """Configurație FastAPI Server"""
    host: str = os.getenv('FASTAPI_HOST', '0.0.0.0')
    port: int = int(os.getenv('FASTAPI_PORT', '8000'))
    debug: bool = os.getenv('FASTAPI_DEBUG', 'false').lower() == 'true'
    reload: bool = os.getenv('FASTAPI_RELOAD', 'false').lower() == 'true'
    
    # Securitate
    secret_key: str = os.getenv('FASTAPI_SECRET_KEY', 'dev-secret-key-CHANGE-IN-PRODUCTION')
    algorithm: str = os.getenv('JWT_ALGORITHM', 'HS256')
    access_token_expire_minutes: int = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
    
    # CORS
    allow_origins: List[str] = field(default_factory=lambda: 
        os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(','))
    allow_credentials: bool = os.getenv('CORS_ALLOW_CREDENTIALS', 'true').lower() == 'true'
    allow_methods: List[str] = field(default_factory=lambda: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
    allow_headers: List[str] = field(default_factory=lambda: ['*'])
    
    # Rate limiting
    rate_limit_enabled: bool = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    rate_limit_calls: int = int(os.getenv('RATE_LIMIT_CALLS', '100'))
    rate_limit_period: int = int(os.getenv('RATE_LIMIT_PERIOD', '60'))  # secunde
    
    # Logging și monitoring
    log_requests: bool = os.getenv('LOG_REQUESTS', 'true').lower() == 'true'
    enable_metrics: bool = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    
    def validate(self) -> bool:
        """Validează configurația FastAPI"""
        if self.secret_key == 'dev-secret-key-CHANGE-IN-PRODUCTION':
            if os.getenv('ENVIRONMENT') == 'production':
                raise ValueError("Schimbă FASTAPI_SECRET_KEY în producție!")
        return True

@dataclass
class DiscordConfig:
    """Configurație Discord Bot și OAuth"""
    # Bot Token
    bot_token: str = os.getenv('DISCORD_BOT_TOKEN', '')
    
    # OAuth2 pentru Dashboard
    client_id: str = os.getenv('DISCORD_CLIENT_ID', '')
    client_secret: str = os.getenv('DISCORD_CLIENT_SECRET', '')
    redirect_uri: str = os.getenv('DISCORD_REDIRECT_URI', 'http://localhost:8000/auth/discord/callback')
    
    # Bot settings
    command_prefix: str = os.getenv('DISCORD_COMMAND_PREFIX', '!')
    owner_id: Optional[int] = int(os.getenv('DISCORD_OWNER_ID')) if os.getenv('DISCORD_OWNER_ID') else None
    
    # Intents
    intents_message_content: bool = True
    intents_members: bool = True
    intents_guilds: bool = True
    intents_guild_messages: bool = True
    
    # Rate limits și performanță
    max_message_length: int = int(os.getenv('DISCORD_MAX_MESSAGE_LENGTH', '2000'))
    command_cooldown: int = int(os.getenv('DISCORD_COMMAND_COOLDOWN', '3'))
    
    # OAuth Scopes
    oauth_scopes: List[str] = field(default_factory=lambda: ['identify', 'guilds'])
    
    def validate(self) -> bool:
        """Validează configurația Discord"""
        if not self.bot_token:
            raise ValueError("DISCORD_BOT_TOKEN nu este setat!")
        if len(self.bot_token) < 50:
            raise ValueError("DISCORD_BOT_TOKEN pare invalid!")
        
        # Pentru dashboard, verifică OAuth settings
        if not self.client_id or not self.client_secret:
            logging.warning("Discord OAuth nu este configurat complet - dashboard-ul nu va funcționa")
        
        return True

@dataclass
class AIConfig:
    """Configurație pentru modelul AI xlm-roberta-base-toxic-x"""
    # Model settings
    model_name: str = os.getenv('AI_MODEL_NAME', 'xlm-roberta-base-toxic-x')
    use_gpu: bool = os.getenv('AI_USE_GPU', 'false').lower() == 'true'
    device: str = os.getenv('AI_DEVICE', 'auto')  # auto, cpu, cuda, mps
    
    # Processing settings
    max_length: int = int(os.getenv('AI_MAX_LENGTH', '512'))
    batch_size: int = int(os.getenv('AI_BATCH_SIZE', '1'))
    num_workers: int = int(os.getenv('AI_NUM_WORKERS', '1'))
    
    # Cache și performanță
    cache_predictions: bool = os.getenv('AI_CACHE_PREDICTIONS', 'true').lower() == 'true'
    cache_ttl: int = int(os.getenv('AI_CACHE_TTL', '3600'))  # 1 oră
    warm_up_model: bool = os.getenv('AI_WARM_UP', 'true').lower() == 'true'
    
    # Thresholds pentru detectarea toxicității
    default_threshold: float = float(os.getenv('AI_DEFAULT_THRESHOLD', '50.0'))
    strict_threshold: float = float(os.getenv('AI_STRICT_THRESHOLD', '30.0'))
    relaxed_threshold: float = float(os.getenv('AI_RELAXED_THRESHOLD', '70.0'))
    
    # Confidence thresholds
    min_confidence: float = float(os.getenv('AI_MIN_CONFIDENCE', '0.6'))
    high_confidence: float = float(os.getenv('AI_HIGH_CONFIDENCE', '0.9'))

@dataclass
class ModerationConfig:
    """Configurație pentru moderare și acțiuni"""
    
    # Escaladarea automată
    enable_auto_moderation: bool = os.getenv('AUTO_MODERATION', 'true').lower() == 'true'
    auto_delete_toxic: bool = os.getenv('AUTO_DELETE_TOXIC', 'false').lower() == 'true'
    
    # Timpul de timeout-uri (în secunde)
    warning_timeout: int = int(os.getenv('WARNING_TIMEOUT', '0'))  # 0 = doar warning
    first_timeout: int = int(os.getenv('FIRST_TIMEOUT', '3600'))  # 1 oră
    second_timeout: int = int(os.getenv('SECOND_TIMEOUT', '86400'))  # 1 zi
    third_timeout: int = int(os.getenv('THIRD_TIMEOUT', '604800'))  # 1 săptămână
    ban_duration: int = int(os.getenv('BAN_DURATION', '2592000'))  # 30 zile
    
    # Limite
    max_warnings: int = int(os.getenv('MAX_WARNINGS', '3'))
    max_timeout_duration: int = int(os.getenv('MAX_TIMEOUT_DURATION', '2419200'))  # 28 zile
    
    # Escalation steps (personalizabile)
    escalation_steps: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"warnings": 1, "action": "dm_warning", "duration": 0},
        {"warnings": 2, "action": "public_warning", "duration": 0},
        {"warnings": 3, "action": "timeout", "duration": 3600},
        {"warnings": 5, "action": "timeout", "duration": 86400},
        {"warnings": 7, "action": "ban", "duration": 604800}
    ])
    
    # Categorii și severitate
    severity_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'hate_speech': 2.0,
        'threat': 2.5,
        'harassment': 1.5,
        'discrimination': 2.0,
        'profanity': 1.0,
        'spam': 0.8
    })

@dataclass
class ReactConfig:
    """Configurație pentru frontend React"""
    # Build și development
    build_path: str = os.getenv('REACT_BUILD_PATH', 'frontend/dist')
    public_path: str = os.getenv('REACT_PUBLIC_PATH', '/static')
    
    # API endpoints
    api_base_url: str = os.getenv('REACT_API_BASE_URL', 'http://localhost:8000/api')
    websocket_url: str = os.getenv('REACT_WEBSOCKET_URL', 'ws://localhost:8000/ws')
    
    # Feature flags
    enable_realtime: bool = os.getenv('REACT_ENABLE_REALTIME', 'true').lower() == 'true'
    enable_charts: bool = os.getenv('REACT_ENABLE_CHARTS', 'true').lower() == 'true'
    enable_exports: bool = os.getenv('REACT_ENABLE_EXPORTS', 'true').lower() == 'true'
    
    # Theme și UI
    default_theme: str = os.getenv('REACT_DEFAULT_THEME', 'light')
    enable_dark_mode: bool = os.getenv('REACT_ENABLE_DARK_MODE', 'true').lower() == 'true'

@dataclass
class LoggingConfig:
    """Configurație pentru logging"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = os.getenv('LOG_FORMAT', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File logging
    log_to_file: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    file_path: str = os.getenv('LOG_FILE_PATH', 'logs/app.log')
    max_file_size: int = int(os.getenv('LOG_MAX_FILE_SIZE', '10485760'))  # 10MB
    backup_count: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    # Component-specific logging
    discord_log_level: str = os.getenv('DISCORD_LOG_LEVEL', 'WARNING')
    transformers_log_level: str = os.getenv('TRANSFORMERS_LOG_LEVEL', 'WARNING')
    fastapi_log_level: str = os.getenv('FASTAPI_LOG_LEVEL', 'INFO')
    mongodb_log_level: str = os.getenv('MONGODB_LOG_LEVEL', 'WARNING')

class Config:
    """Clasa principală de configurație"""
    
    def __init__(self):
        # Încarcă toate configurațiile
        self.mongodb = MongoDBConfig()
        self.redis = RedisConfig()
        self.fastapi = FastAPIConfig()
        self.discord = DiscordConfig()
        self.ai = AIConfig()
        self.moderation = ModerationConfig()
        self.react = ReactConfig()
        self.logging = LoggingConfig()
        
        # Metadata aplicație
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.version = os.getenv('APP_VERSION', '1.0.0')
        self.app_name = os.getenv('APP_NAME', 'Discord Anti-Toxicity Bot')
        
        # Încarcă configurația din JSON dacă există
        self._load_json_config()
        
        # Validează configurația
        self._validate_config()
        
        # Setup logging
        self.setup_logging()
    
    def _load_json_config(self):
        """Încarcă configurația din config.json"""
        config_files = [
            os.getenv('CONFIG_FILE', 'config.json'),
            'config.local.json',  # Pentru override-uri locale
            f'config.{self.environment}.json'  # Pentru environment-specific
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        json_config = json.load(f)
                    
                    self._apply_json_config(json_config)
                    logging.info(f"✅ Configurație încărcată din {config_file}")
                    
                except Exception as e:
                    logging.warning(f"⚠️ Eroare la încărcarea {config_file}: {e}")
    
    def _apply_json_config(self, json_config: Dict[str, Any]):
        """Aplică configurația din JSON"""
        # AI thresholds
        if 'ai_settings' in json_config:
            ai_settings = json_config['ai_settings']
            for key, value in ai_settings.items():
                if hasattr(self.ai, key):
                    setattr(self.ai, key, value)
        
        # Moderation escalation
        if 'moderation_settings' in json_config:
            mod_settings = json_config['moderation_settings']
            for key, value in mod_settings.items():
                if hasattr(self.moderation, key):
                    setattr(self.moderation, key, value)
        
        # Server-specific overrides
        if 'server_overrides' in json_config:
            self.server_overrides = json_config['server_overrides']
        
        # Feature flags
        if 'features' in json_config:
            self.feature_flags = json_config['features']
    
    def _validate_config(self):
        """Validează toate configurațiile"""
        errors = []
        warnings = []
        
        try:
            # Validează componentele critice
            self.discord.validate()
            self.fastapi.validate()
            
            # Verificări pentru producție
            if self.is_production():
                if 'dev-secret-key' in self.fastapi.secret_key:
                    errors.append("FastAPI: Schimbă secret_key în producție!")
                
                if self.debug:
                    warnings.append("Debug mode este activat în producție")
                
                if not self.mongodb.username or not self.mongodb.password:
                    warnings.append("MongoDB: Nu ai credențiale în producție")
            
            # Verifică dependențele
            if self.ai.use_gpu:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("AI: GPU cerut dar nu este disponibil")
        
        except Exception as e:
            errors.append(f"Validare generală: {e}")
        
        # Raportează rezultatele
        if errors:
            print("❌ ERORI CRITICE DE CONFIGURAȚIE:")
            for error in errors:
                print(f"   - {error}")
            raise RuntimeError("Configurația conține erori critice!")
        
        if warnings:
            print("⚠️ AVERTISMENTE CONFIGURAȚIE:")
            for warning in warnings:
                print(f"   - {warning}")
    
    def setup_logging(self):
        """Configurează logging-ul global"""
        # Creează directorul pentru log-uri
        if self.logging.log_to_file:
            log_dir = os.path.dirname(self.logging.file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
        
        # Configurează handler-ele
        handlers = [logging.StreamHandler()]
        
        if self.logging.log_to_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            handlers.append(file_handler)
        
        # Setup logging principal
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=handlers
        )
        
        # Configurează logging pentru librării externe
        logging.getLogger('discord').setLevel(
            getattr(logging, self.logging.discord_log_level.upper())
        )
        logging.getLogger('transformers').setLevel(
            getattr(logging, self.logging.transformers_log_level.upper())
        )
        logging.getLogger('motor').setLevel(
            getattr(logging, self.logging.mongodb_log_level.upper())
        )
    
    def is_development(self) -> bool:
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        return self.environment == 'production'
    
    def is_testing(self) -> bool:
        return self.environment == 'testing'
    
    def get_mongodb_url(self) -> str:
        """Returnează URL-ul MongoDB"""
        return self.mongodb.connection_string
    
    def get_server_config(self, server_id: int) -> Dict[str, Any]:
        """Obține configurația pentru un server specific"""
        base_config = {
            'toxicity_threshold': self.ai.default_threshold,
            'auto_moderation': self.moderation.enable_auto_moderation,
            'max_warnings': self.moderation.max_warnings,
            'escalation_steps': self.moderation.escalation_steps
        }
        
        # Aplică override-uri server-specific dacă există
        if hasattr(self, 'server_overrides') and str(server_id) in self.server_overrides:
            base_config.update(self.server_overrides[str(server_id)])
        
        return base_config
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convertește configurația la dicționar pentru debugging"""
        config_dict = {
            'environment': self.environment,
            'version': self.version,
            'app_name': self.app_name,
            'debug': self.debug,
            'mongodb': {
                'host': self.mongodb.host,
                'port': self.mongodb.port,
                'database': self.mongodb.database,
                'use_ssl': self.mongodb.use_ssl
            },
            'fastapi': {
                'host': self.fastapi.host,
                'port': self.fastapi.port,
                'debug': self.fastapi.debug,
                'rate_limit_enabled': self.fastapi.rate_limit_enabled
            },
            'ai': {
                'model_name': self.ai.model_name,
                'use_gpu': self.ai.use_gpu,
                'default_threshold': self.ai.default_threshold,
                'cache_predictions': self.ai.cache_predictions
            },
            'discord': {
                'command_prefix': self.discord.command_prefix,
                'max_message_length': self.discord.max_message_length
            }
        }
        
        if include_secrets:
            config_dict['secrets'] = {
                'mongodb_has_auth': bool(self.mongodb.username),
                'discord_bot_token_set': bool(self.discord.bot_token),
                'discord_oauth_configured': bool(self.discord.client_id and self.discord.client_secret),
                'fastapi_secret_set': bool(self.fastapi.secret_key)
            }
        
        return config_dict

# Instanță globală
config = Config()

# Funcții helper
def get_config() -> Config:
    """Returnează instanța globală de configurație"""
    return config

def get_mongodb_url() -> str:
    """Returnează URL-ul MongoDB"""
    return config.get_mongodb_url()

def get_server_config(server_id: int) -> Dict[str, Any]:
    """Configurația pentru un server specific"""
    return config.get_server_config(server_id)

def is_development() -> bool:
    return config.is_development()

def is_production() -> bool:
    return config.is_production()

if __name__ == "__main__":
    # Test configurația
    print("🔧 === TESTARE CONFIGURAȚIE (MongoDB + FastAPI) ===\n")
    
    try:
        test_config = Config()
        print("✅ Configurația a fost încărcată cu succes!")
        
        print(f"📊 Environment: {test_config.environment}")
        print(f"🍃 MongoDB: {test_config.mongodb.host}:{test_config.mongodb.port}")
        print(f"🚀 FastAPI: {test_config.fastapi.host}:{test_config.fastapi.port}")
        print(f"🤖 AI Model: {test_config.ai.model_name}")
        print(f"🎨 React: {test_config.react.api_base_url}")
        
        # Test configurația completă
        import pprint
        print("\n📋 Configurație completă (fără secrete):")
        pprint.pprint(test_config.to_dict())
        
        print(f"\n🔐 Status secrete:")
        pprint.pprint(test_config.to_dict(include_secrets=True)['secrets'])
        
    except Exception as e:
        print(f"❌ Eroare la testarea configurației: {e}")
        raise