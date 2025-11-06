from .db import db
from datetime import datetime

class HealthRecord(db.Model):
    __tablename__ = 'health_records'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    model_used = db.Column(db.String(50))  # clinical/lifestyle
    risk_score = db.Column(db.Float)
    likely_condition = db.Column(db.String(120))
    recommendation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
