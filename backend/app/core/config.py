from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = "https://your-project.supabase.co"
    supabase_anon_key: str = "your-anon-key"
    supabase_service_role_key: str = "your-service-role-key"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # App security
    secret_key: str = "change-me-to-a-long-random-string"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # CORS — stored as comma-separated string in .env
    cors_origins_raw: str = Field(
        default="http://localhost:5173,https://fl-platform-ui-8vqt.onrender.com,https://fedlearn-os.web.app,https://fedlearn-os.firebaseapp.com",
        validation_alias="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(',') if o.strip()]
    # Environment
    environment: str = "development"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore", "populate_by_name": True}


@lru_cache
def get_settings() -> Settings:
    return Settings()
