from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_query = Column(Text, nullable=False)
    sentiment = Column(String(50), nullable=False) 
    confidence = Column(Float, nullable=True)     
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())