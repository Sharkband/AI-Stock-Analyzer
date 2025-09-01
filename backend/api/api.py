from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import json
from datetime import datetime, timedelta
import os
import pickle
import uvicorn

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # api directory
backend_dir = os.path.dirname(script_dir)  # backend directory
sys.path.insert(0, backend_dir)
print(f"Added to Python path: {backend_dir}")


# Import your models (assuming they're in stock_models.py)
try:
    from models.stock_models import StockAnalyzer, LSTMStockPredictor, TransformerStockPredictor
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import torch
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure stock_models.py is in the same directory and all packages are installed")

# FastAPI app initialization
app = FastAPI(
    title="AI Stock Analyzer API",
    description="PyTorch-based stock analysis and prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
trained_models = {}
training_status = {}

# Pydantic models for request/response
class TrainingRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    model_type: str = Field(default="lstm", description="Model type: lstm, transformer, cnn_lstm")
    sequence_length: int = Field(default=60, description="Input sequence length")
    epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size for training")

class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    days_ahead: int = Field(default=1, description="Days to predict ahead")
    model_type: str = Field(default="lstm", description="Model type to use")

class StockDataRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    period: str = Field(default="1y", description="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")

class TrainingResponse(BaseModel):
    message: str
    task_id: str
    symbol: str
    model_type: str
    status: str

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    percent_change: float
    confidence: float
    model_type: str
    timestamp: str
    last_updated : str

class StockDataResponse(BaseModel):
    symbol: str
    data: List[Dict[str, Any]]
    period: str
    total_records: int

class ModelInfo(BaseModel):
    symbol: str
    model_type: str
    trained_at: str
    train_rmse: float
    val_rmse: float
    status: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    gpu_available: bool

# Utility functions
def generate_task_id() -> str:
    """Generate unique task ID for training jobs"""
    return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

def save_model(analyzer: StockAnalyzer, symbol: str, model_type: str) -> str:
    """Save trained model to disk"""
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = f"{model_dir}/{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
    
    model_data = {
        'analyzer': analyzer,
        'symbol': symbol,
        'model_type': model_type,
        'trained_at': datetime.now().isoformat(),
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_path

def load_model(symbol: str, model_type: str) -> Optional[StockAnalyzer]:
    """Load trained model from disk"""
    model_dir = "saved_models"
    
    # Look for the most recent model file
    pattern = f"{symbol}_{model_type}_"
    model_files = []
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith(pattern) and file.endswith('.pkl'):
                model_files.append(file)
    
    if not model_files:
        return None
    
    # Load the most recent model
    latest_file = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_file)
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['analyzer']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Stock Analyzer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(trained_models),
        gpu_available=torch.cuda.is_available()
    )

@app.get("/stock-data/{symbol}", response_model=StockDataResponse)
async def get_stock_data(symbol: str, period: str = "1y"):
    """Fetch historical stock data"""
    try:
        stock = yf.Ticker(symbol.upper())
        df = stock.history(period=period)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Convert DataFrame to list of dictionaries
        data = []
        for index, row in df.iterrows():
            data.append({
                "date": index.strftime('%Y-%m-%d'),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        return StockDataResponse(
            symbol=symbol.upper(),
            data=data,
            period=period,
            total_records=len(data)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training in background"""
    task_id = generate_task_id()
    symbol = request.symbol.upper()
    
    # Check if model is already training
    if f"{symbol}_{request.model_type}" in training_status:
        current_status = training_status[f"{symbol}_{request.model_type}"]
        if current_status["status"] == "training":
            raise HTTPException(status_code=409, detail="Model is already training")
    
    # Initialize training status
    training_status[f"{symbol}_{request.model_type}"] = {
        "task_id": task_id,
        "status": "training",
        "started_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Start training in background
    background_tasks.add_task(
        train_model_background,
        task_id,
        symbol,
        request.model_type,
        request.sequence_length,
        request.epochs,
        request.batch_size
    )
    
    return TrainingResponse(
        message="Training started",
        task_id=task_id,
        symbol=symbol,
        model_type=request.model_type,
        status="training"
    )

async def train_model_background(task_id: str, symbol: str, model_type: str, 
                               sequence_length: int, epochs: int, batch_size: int):
    """Background task for model training"""
    try:
        print(f"Starting training for {symbol} with {model_type} model...")
        
        # Initialize analyzer
        analyzer = StockAnalyzer(model_type=model_type, sequence_length=sequence_length)
        
        # Train model
        results = analyzer.train_on_stock(symbol)
        
        if results:
            # Save trained model
            model_path = save_model(analyzer, symbol, model_type)
            
            # Store in memory for quick access
            trained_models[f"{symbol}_{model_type}"] = {
                "analyzer": analyzer,
                "results": results,
                "trained_at": datetime.now().isoformat(),
                "model_path": model_path
            }
            
            # Update status
            training_status[f"{symbol}_{model_type}"] = {
                "task_id": task_id,
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "train_rmse": results["train_rmse"],
                "val_rmse": results["val_rmse"]
            }
            
            print(f"Training completed for {symbol}")
        else:
            training_status[f"{symbol}_{model_type}"]["status"] = "failed"
            print(f"Training failed for {symbol}")
    
    except Exception as e:
        training_status[f"{symbol}_{model_type}"]["status"] = "failed"
        training_status[f"{symbol}_{model_type}"]["error"] = str(e)
        print(f"Training error for {symbol}: {e}")

@app.get("/training-status/{symbol}/{model_type}")
async def get_training_status(symbol: str, model_type: str):
    """Get training status for a specific model"""
    key = f"{symbol.upper()}_{model_type}"
    
    if key not in training_status:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_status[key]

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """Predict future stock price"""
    symbol = request.symbol.upper()
    model_key = f"{symbol}_{request.model_type}"
    
    
    try:
        # Try to get model from memory first
        analyzer = None
        if model_key in trained_models:
            analyzer = trained_models[model_key]["analyzer"]
        else:
            # Try to load from disk
            analyzer = load_model(symbol, request.model_type)
        
        if analyzer is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No trained model found for {symbol} with {request.model_type}. Please train the model first."
            )
        
        # Make prediction
        prediction = analyzer.predict_next_price(symbol, request.days_ahead)
        print(request.days_ahead)
        print(prediction["predicted_price"])

        return PredictionResponse(
            symbol=symbol,
            current_price=prediction["current_price"],
            predicted_price=prediction["predicted_price"],
            price_change=prediction["price_change"],
            percent_change=prediction["percent_change"],
            confidence=prediction["confidence"],
            model_type=request.model_type,
            timestamp=datetime.now().isoformat(),
            last_updated = prediction["last_updated"].strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models", response_model=List[ModelInfo])
async def list_trained_models():
    """List all trained models"""
    models = []
    
    # Models in memory
    for key, model_data in trained_models.items():
        symbol, model_type = key.split('_', 1)
        models.append(ModelInfo(
            symbol=symbol,
            model_type=model_type,
            trained_at=model_data["trained_at"],
            train_rmse=model_data["results"]["train_rmse"],
            val_rmse=model_data["results"]["val_rmse"],
            status="loaded"
        ))
    
    # Models on disk (not in memory)
    model_dir = "saved_models"
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                try:
                    parts = file.replace('.pkl', '').split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        model_type = '_'.join(parts[1:-1])
                        key = f"{symbol}_{model_type}"
                        
                        if key not in trained_models:  # Not already listed
                            with open(os.path.join(model_dir, file), 'rb') as f:
                                model_data = pickle.load(f)
                            
                            models.append(ModelInfo(
                                symbol=symbol,
                                model_type=model_type,
                                trained_at=model_data["trained_at"],
                                train_rmse=0.0,  # Not available without loading
                                val_rmse=0.0,   # Not available without loading
                                status="saved"
                            ))
                except Exception as e:
                    print(f"Error reading model file {file}: {e}")
    
    return models

@app.delete("/models/{symbol}/{model_type}")
async def delete_model(symbol: str, model_type: str):
    """Delete a trained model"""
    symbol = symbol.upper()
    model_key = f"{symbol}_{model_type}"
    
    # Remove from memory
    if model_key in trained_models:
        del trained_models[model_key]
    
    # Remove from disk
    model_dir = "saved_models"
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith(f"{symbol}_{model_type}_") and file.endswith('.pkl'):
                try:
                    os.remove(os.path.join(model_dir, file))
                    print(f"Deleted model file: {file}")
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")
    
    return {"message": f"Model {symbol}_{model_type} deleted successfully"}

@app.get("/market-overview")
async def market_overview():
    """Get overview of major market indices"""
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT"
    }
    
    overview = {}
    
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2d")  # Get last 2 days
            
            if len(data) >= 2:
                current = data['Close'].iloc[-1]
                previous = data['Close'].iloc[-2]
                change = current - previous
                change_percent = (change / previous) * 100
                
                overview[name] = {
                    "current_price": round(current, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "symbol": symbol
                }
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            overview[name] = {"error": str(e)}
    
    return overview

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 AI Stock Analyzer API starting up...")
    print(f"📊 GPU Available: {torch.cuda.is_available()}")
    
    # Create necessary directories
    os.makedirs("saved_models", exist_ok=True)
    
    print("✅ API ready to serve requests!")

# Main function to run the server
def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "api.api:app",  # Assuming this file is named api.py
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes during development
        log_level="info"
    )

if __name__ == "__main__":
    main()