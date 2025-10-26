from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    ENVIRONMENT: str = "development"
    
    API_KEY: str = ""
    CORS_ORIGINS: str = "*"
    
    MODEL_NAME: str = "d4data/bias-detection-model"
    MODEL_CACHE_DIR: str = "./model_cache"
    
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to prevent re-reading the .env file on every call
    """
    return Settings()