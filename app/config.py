from pydantic import BaseSettings

class Settings(BaseSettings):
    database_hostname: str
    database_name: str
    database_user: str
    database_password: str
    database_connstring: str

    class Config:
        env_file = ".env"

settings = Settings()