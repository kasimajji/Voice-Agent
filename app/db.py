import os
import time
import logging
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

logger = logging.getLogger("voice_agent.db")


def get_database_url() -> str:
    """
    Build DATABASE_URL from environment variables.
    Supports both explicit DATABASE_URL and individual MySQL credentials.
    """
    # Check for explicit DATABASE_URL first
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        logger.info("Using DATABASE_URL from environment")
        return database_url
    
    # Build MySQL URL from individual components
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "voice_ai")
    
    # Use pymysql driver for MySQL
    # URL-encode password to handle special characters like @, :, /
    mysql_url = f"mysql+pymysql://{db_user}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"
    logger.info(f"Using MySQL at {db_host}:{db_port}/{db_name}")
    return mysql_url


def create_engine_with_retry(database_url: str, max_retries: int = 5, retry_delay: int = 2):
    """
    Create database engine with retry logic for MySQL readiness.
    This is crucial for Docker Compose where the app may start before MySQL is ready.
    Supports SQLite for testing (no connect_args needed).
    """
    is_sqlite = database_url.startswith("sqlite")
    
    if is_sqlite:
        connect_args = {}
    else:
        connect_args = {
            "connect_timeout": 10,
            "charset": "utf8mb4"
        }
    
    engine = create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=not is_sqlite,  # Verify connections before using (not needed for SQLite)
        pool_recycle=3600,   # Recycle connections after 1 hour
        echo=False
    )
    
    # Test connection with retries
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Connection attempt {attempt}/{max_retries}...")
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to database")
            return engine
        except OperationalError as e:
            if attempt < max_retries:
                logger.warning(f"Connection failed: {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                raise
    
    return engine


# Initialize database connection
DATABASE_URL = get_database_url()
engine = create_engine_with_retry(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
