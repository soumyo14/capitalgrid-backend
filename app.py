import os
# Force CPU to prevent memory fragmentation during model loading
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ---------------- LIFESPAN (Global Model Loading) ---------------- #

MODELS = {}
BANK_LIST = ["AXISBANK", "BANKBARODA", "CANBK", "HDFCBANK", "ICICIBANK", 
             "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB", "SBIN"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once into memory when the server starts
    print("🚀 CapitalGrid AI: Loading Models...")
    for bank in BANK_LIST:
        try:
            model_path = f"backend/models/model_{bank}.h5"
            if os.path.exists(model_path):
                MODELS[bank] = load_model(model_path)
                print(f"✅ Loaded {bank}")
            else:
                print(f"⚠️ Model for {bank} not found at {model_path}")
        except Exception as e:
            print(f"❌ Error loading {bank}: {e}")
    yield
    # Clean up on shutdown
    MODELS.clear()
    print("🛑 CapitalGrid AI: Shutting down...")

# ---------------- APP SETUP ---------------- #

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HELPERS ---------------- #

def get_bank_model(bank: str):
    return MODELS.get(bank)

# ---------------- ROUTES ---------------- #

@app.get("/")
def root():
    return {"message": "CapitalGrid AI Prediction API is active"}

@app.get("/predict")
def predict(bank: str):
    ticker = bank + ".NS"
    
    # Use 2 years to anchor the scaler so prices remain realistic
    df_anchor = yf.download(ticker, period="2y", progress=False)
    index_anchor = yf.download("^NSEBANK", period="2y", progress=False)

    if df_anchor.empty or len(df_anchor) < 60:
        return {"error": "Not enough data for prediction"}

    # Flatten columns for single-level access
    for d in [df_anchor, index_anchor]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    df = df_anchor[['Close']].copy()
    index = index_anchor[['Close']].copy()
    index.rename(columns={'Close': 'NIFTY_Close'}, inplace=True)

    df = df.merge(index, left_index=True, right_index=True, how="left")
    df['NIFTY_Close'] = df['NIFTY_Close'].ffill()

    # Fit scaler on long history to stabilize the prediction range
    scaler = MinMaxScaler()
    scaler.fit(df) 
    
    recent_data = df.tail(60)
    scaled_input = scaler.transform(recent_data)
    X = np.array([scaled_input])

    model = get_bank_model(bank)
    if not model:
        return {"error": f"Model for {bank} not loaded"}

    pred = model.predict(X, verbose=0)[0][0]

    # Inverse scale correctly using a dummy array
    temp = np.zeros((1, 2))
    temp[0][0] = pred
    price = scaler.inverse_transform(temp)[0][0]

    return {"bank": bank, "predicted_price": float(price)}

@app.get("/history")
def history(bank: str, period: str = "5y"):
    ticker = bank + ".NS"
    df = yf.download(ticker, period=period, progress=False)

    if df.empty:
        return {"error": "No data found"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    close = pd.Series(df["Close"]).astype(float)

    return {
        "bank": bank,
        "dates": df["Date"].astype(str).tolist(),
        "close": close.tolist()
    }

# ---------------- TRADE LOGIC ---------------- #

def save_trade(trade: dict):
    path = "backend/trades.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
                
    data.append(trade)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

@app.post("/trade/buy")
def buy(bank: str, quantity: int, price: float):
    trade = {
        "bank": bank,
        "quantity": quantity,
        "price": price,
        "type": "BUY",
        "time": datetime.now().isoformat()
    }
    save_trade(trade)
    return {"status": "success", "trade": trade}

@app.post("/trade/sell")
def sell(bank: str, quantity: int, price: float):
    trade = {
        "bank": bank,
        "quantity": quantity,
        "price": price,
        "type": "SELL",
        "time": datetime.now().isoformat()
    }
    save_trade(trade)
    return {"status": "success", "trade": trade}

@app.get("/portfolio")
def portfolio():
    path = "backend/trades.json"
    if not os.path.exists(path):
        return {"trades": [], "holdings": {}}

    with open(path, "r") as f:
        trades = json.load(f)

    holdings = {}
    for t in trades:
        qty = t["quantity"] if t["type"] == "BUY" else -t["quantity"]
        holdings[t["bank"]] = holdings.get(t["bank"], 0) + qty

    return {"trades": trades, "holdings": holdings}