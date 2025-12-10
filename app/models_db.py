from datetime import datetime

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    dog_images = relationship("DogImage", back_populates="user")
    generated_images = relationship("GeneratedImage", back_populates="user")


class DogImage(Base):
    __tablename__ = "dog_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_path = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="dog_images")
    generated_images = relationship("GeneratedImage", back_populates="dog_image")


class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dog_image_id = Column(Integer, ForeignKey("dog_images.id"), nullable=False)
    file_path = Column(String(512), nullable=False)
    model_version = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="generated_images")
    dog_image = relationship("DogImage", back_populates="generated_images")
