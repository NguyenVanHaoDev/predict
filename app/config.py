import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "My Project")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
settings = Settings()