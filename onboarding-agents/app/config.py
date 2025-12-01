from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://user:pass@localhost:5432/onboarding"
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str | None = None
    class Config:
        env_file = ".env"
settings = Settings()
