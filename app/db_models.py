from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

from .database import Base


class Generation(Base):
    __tablename__ = "generations"

    id = Column(Integer, primary_key=True, index=True)
    input_path = Column(String, nullable=False)
    output_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
