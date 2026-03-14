from sqlalchemy import create_engine

from config import settings


def get_engine():
    return create_engine(settings.db_url)
