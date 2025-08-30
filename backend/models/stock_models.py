import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from typing import Tuple, Optional
import math

class LSTMStockPredictor(nn.Module):
    """
    LSTM-based model for stock price prediction
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1):
        super(LSTMStockPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        out = self.dropout(context_vector)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

class TransformerStockPredictor(nn.Module):
    """
    Transformer-based model for stock analysis
    """
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1, output_size: int = 1):
        super(TransformerStockPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use the last time step for prediction
        x = x[:, -1, :]
        
        # Final projection
        output = self.output_projection(x)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class CNNLSTMStockPredictor(nn.Module):
    """
    CNN-LSTM hybrid model for capturing both local patterns and temporal dependencies
    """
    def __init__(self, input_size: int, cnn_filters: int = 64, kernel_size: int = 3,
                 lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2):
        super(CNNLSTMStockPredictor, self).__init__()
        
        # CNN layers for local pattern extraction
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)
        
    def forward(self, x):
        # CNN feature extraction
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        
        return out

class StockDataProcessor:
    """
    Data preprocessing and feature engineering for stock data
    """
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_change'] = df['Close'].pct_change()
        df['Price_momentum_5'] = df['Close'].pct_change(periods=5)
        df['Price_momentum_10'] = df['Close'].pct_change(periods=10)
        
        # Volatility
        df['Volatility'] = df['Price_change'].rolling(window=20).std()
        
        return df
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and prepare features for training"""
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_position', 'Volume_ratio',
            'Price_change', 'Price_momentum_5', 'Price_momentum_10',
            'Volatility'
        ]
        
        # Store feature columns for later use
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        # Select features and drop NaN values
        features_df = df[self.feature_columns].dropna()
        
        return features_df
        
    def create_sequences(self, data: np.ndarray, target_col_idx: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_col_idx])  # Close price
            
        return np.array(X), np.array(y)
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
        """Complete preprocessing pipeline"""
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(features_df.values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        return X_tensor, y_tensor, self.scaler

class StockTrainer:
    """
    Training pipeline for stock prediction models
    """
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_val: torch.Tensor, y_val: torch.Tensor,
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the model"""
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
            return predictions.cpu().numpy()

class MultiTaskStockModel(nn.Module):
    """
    Multi-task model that predicts both price and trend direction
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(MultiTaskStockModel, self).__init__()
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Price regression head
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Trend classification head (up/down/sideways)
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3),  # 3 classes: up, down, sideways
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Shared representation
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Task-specific predictions
        price_pred = self.price_head(last_hidden)
        trend_pred = self.trend_head(last_hidden)
        
        return price_pred, trend_pred

class StockAnalyzer:
    """
    Main class for stock analysis and prediction
    """
    def __init__(self, model_type: str = 'lstm', sequence_length: int = 60):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.processor = StockDataProcessor(sequence_length)
        self.model = None
        self.trainer = None
        
    def fetch_stock_data(self, symbol: str, period: str = '2y') -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
            
    def prepare_model(self, input_size: int):
        """Initialize the model based on type"""
        if self.model_type == 'lstm':
            self.model = LSTMStockPredictor(input_size=input_size)
        elif self.model_type == 'transformer':
            self.model = TransformerStockPredictor(input_size=input_size)
        elif self.model_type == 'cnn_lstm':
            self.model = CNNLSTMStockPredictor(input_size=input_size)
        elif self.model_type == 'multitask':
            self.model = MultiTaskStockModel(input_size=input_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.trainer = StockTrainer(self.model)
        
    def train_on_stock(self, symbol: str, train_split: float = 0.8):
        """Complete training pipeline for a stock"""
        print(f"Training model on {symbol}...")
        
        # Fetch and preprocess data
        df = self.fetch_stock_data(symbol)
        if df is None:
            return None
            
        X, y, scaler = self.processor.preprocess_data(df)
        
        # Train-validation split
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Initialize model
        input_size = X.shape[2]
        self.prepare_model(input_size)
        
        # Train model
        self.trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate
        train_pred = self.trainer.predict(X_train)
        val_pred = self.trainer.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train.numpy(), train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val.numpy(), val_pred))
        
        print(f"Training RMSE: {train_rmse:.6f}")
        print(f"Validation RMSE: {val_rmse:.6f}")
        
        return {
            'model': self.model,
            'scaler': scaler,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'feature_columns': self.processor.feature_columns
        }
        
    def predict_next_price(self, symbol: str, days_ahead: int = 1) -> dict:
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_on_stock first.")
            
        # Fetch recent data
        df = self.fetch_stock_data(symbol, period='3mo')
        df = self.processor.add_technical_indicators(df)
        features_df = self.processor.prepare_features(df)
        
        # Use last sequence for prediction
        last_sequence = features_df.values[-self.sequence_length:]
        last_sequence_scaled = self.processor.scaler.transform(last_sequence)
        
        # Convert to tensor
        X_pred = torch.FloatTensor(last_sequence_scaled).unsqueeze(0)
        
        # Make prediction
        prediction_scaled = self.trainer.predict(X_pred)
        
        # Inverse transform (approximate)
        # Note: This assumes Close price is at index 3 in features
        dummy_array = np.zeros((1, len(self.processor.feature_columns)))
        dummy_array[0, 3] = prediction_scaled[0, 0]
        prediction = self.processor.scaler.inverse_transform(dummy_array)[0, 3]
        
        current_price = df['Close'].iloc[-1]
        price_change = prediction - current_price
        percent_change = (price_change / current_price) * 100
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': prediction,
            'price_change': price_change,
            'percent_change': percent_change,
            'confidence': self.calculate_confidence(df)
        }
        
    def calculate_confidence(self, df: pd.DataFrame) -> float:
        """Calculate prediction confidence based on recent volatility"""
        recent_volatility = df['Close'].pct_change().tail(20).std()
        # Lower volatility = higher confidence
        confidence = max(0.1, min(0.9, 1 - (recent_volatility * 10)))
        return confidence

# Example usage and training script
def main():
    """Example usage of the stock analyzer"""
    
    # Initialize analyzer
    analyzer = StockAnalyzer(model_type='lstm', sequence_length=60)
    
    # Train on a stock (e.g., Apple)
    results = analyzer.train_on_stock('AAPL')
    
    if results:
        print("Training completed successfully!")
        
        # Make predictions
        prediction = analyzer.predict_next_price('AAPL')
        print(f"\nPrediction for AAPL:")
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
        print(f"Expected Change: ${prediction['price_change']:.2f} ({prediction['percent_change']:.2f}%)")
        print(f"Confidence: {prediction['confidence']:.2f}")

if __name__ == "__main__":
    main()