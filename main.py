from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, validator
from datetime import datetime, timedelta
from typing import List, Optional
import hashlib
import jwt
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import logging
import os
from dotenv import load_dotenv
import os
load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/banking_db")
SECRET_KEY = os.getenv("SECRET_KEY", "change")
# Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
app = FastAPI(title="Banking API", version="1.0.0")
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    accounts = relationship("Account", back_populates="owner")

class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    account_number = Column(String, unique=True, index=True)
    account_type = Column(String)
    balance = Column(Float, default=0.0)
    currency = Column(String, default="USD")
    is_frozen = Column(Boolean, default=False)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    transaction_type = Column(String)
    amount = Column(Float)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_flagged = Column(Boolean, default=False)
    fraud_score = Column(Float, default=0.0)
    account_id = Column(Integer, ForeignKey("accounts.id"))
    account = relationship("Account", back_populates="transactions")

# Pydantic Schemas
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class AccountCreate(BaseModel):
    account_type: str
    currency: str = "USD"

class TransactionCreate(BaseModel):
    transaction_type: str
    amount: float
    description: str = ""
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class TransferRequest(BaseModel):
    from_account_id: int
    to_account_id: int
    amount: float
    description: str = ""

# Utilities
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token(user_id: int) -> str:
    payload = {"user_id": user_id, "exp": datetime.utcnow() + timedelta(hours=24)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload.get("user_id")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_account_number() -> str:
    import random
    return f"ACC{random.randint(1000000000, 9999999999)}"

# Fraud Detection (Simplified)
class FraudDetector:
    def detect_fraud(self, amount: float, user_transactions: List[Transaction]) -> tuple:
        if not user_transactions:
            return False, 0.0
        
        amounts = [txn.amount for txn in user_transactions[-20:]]
        avg_amount = np.mean(amounts)
        std_amount = np.std(amounts) if len(amounts) > 1 else avg_amount
        
        # Simple rule: flag if > 3 std deviations or > $10k
        if std_amount > 0 and abs(amount - avg_amount) / std_amount > 3:
            return True, 0.8
        if amount > 10000:
            return True, 0.9
        return False, 0.1

fraud_detector = FraudDetector()

# API Routes
@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = User(email=user.email, hashed_password=hash_password(user.password), full_name=user.full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"id": db_user.id, "email": db_user.email}

@app.post("/login")
async def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or hash_password(password) != user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(user.id), "token_type": "bearer"}

@app.post("/accounts")
async def create_account(account: AccountCreate, user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    db_account = Account(account_number=generate_account_number(), account_type=account.account_type, 
                        currency=account.currency, owner_id=user_id)
    db.add(db_account)
    db.commit()
    db.refresh(db_account)
    return {"id": db_account.id, "account_number": db_account.account_number, "balance": db_account.balance}

@app.get("/accounts")
async def get_accounts(user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    accounts = db.query(Account).filter(Account.owner_id == user_id).all()
    return [{"id": acc.id, "account_number": acc.account_number, "balance": acc.balance, "type": acc.account_type} for acc in accounts]

@app.post("/transactions")
async def create_transaction(transaction: TransactionCreate, account_id: int, user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    account = db.query(Account).filter(Account.id == account_id, Account.owner_id == user_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    if account.is_frozen:
        raise HTTPException(status_code=400, detail="Account is frozen")
    if transaction.transaction_type == "withdrawal" and account.balance < transaction.amount:
        raise HTTPException(status_code=400, detail="Insufficient funds")
    
    # Fraud detection
    user_transactions = db.query(Transaction).filter(Transaction.account_id == account_id).all()
    is_fraud, fraud_score = fraud_detector.detect_fraud(transaction.amount, user_transactions)
    
    # Create transaction and update balance
    db_transaction = Transaction(transaction_type=transaction.transaction_type, amount=transaction.amount,
                               description=transaction.description, account_id=account_id, 
                               is_flagged=is_fraud, fraud_score=fraud_score)
    
    if transaction.transaction_type == "deposit":
        account.balance += transaction.amount
    elif transaction.transaction_type == "withdrawal":
        account.balance -= transaction.amount
    
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    
    return {"id": db_transaction.id, "amount": db_transaction.amount, "fraud_flagged": is_fraud, 
            "new_balance": account.balance}

@app.post("/transfer")
async def transfer_funds(transfer: TransferRequest, user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    from_account = db.query(Account).filter(Account.id == transfer.from_account_id, Account.owner_id == user_id).first()
    to_account = db.query(Account).filter(Account.id == transfer.to_account_id).first()
    
    if not from_account or not to_account:
        raise HTTPException(status_code=404, detail="Account not found")
    if from_account.balance < transfer.amount:
        raise HTTPException(status_code=400, detail="Insufficient funds")
    
    # Update balances and create transactions
    from_account.balance -= transfer.amount
    to_account.balance += transfer.amount
    
    withdrawal = Transaction(transaction_type="withdrawal", amount=transfer.amount, 
                           description=f"Transfer to {to_account.account_number}", account_id=from_account.id)
    deposit = Transaction(transaction_type="deposit", amount=transfer.amount,
                        description=f"Transfer from {from_account.account_number}", account_id=to_account.id)
    
    db.add_all([withdrawal, deposit])
    db.commit()
    
    return {"message": "Transfer completed", "from_balance": from_account.balance, "to_balance": to_account.balance}

@app.get("/transactions/{account_id}")
async def get_transactions(account_id: int, user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    account = db.query(Account).filter(Account.id == account_id, Account.owner_id == user_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    transactions = db.query(Transaction).filter(Transaction.account_id == account_id).all()
    return [{"id": t.id, "type": t.transaction_type, "amount": t.amount, "description": t.description, 
             "timestamp": t.timestamp, "fraud_flagged": t.is_flagged} for t in transactions]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Initialize database
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Banking API")