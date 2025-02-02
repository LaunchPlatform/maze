from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from ..core.config import settings

engine = create_engine(str(settings.DATABASE_URL), pool_pre_ping=True)


SessionMaker = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
Session = scoped_session(SessionMaker)
