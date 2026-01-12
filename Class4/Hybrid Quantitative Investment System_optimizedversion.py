# Install Packages
!pip install yfinance pandas numpy scipy scikit-learn ta xgboost lightgbm matplotlib seaborn reportlab optuna -q
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import json
from pathlib import Path
import joblib
import optuna
import lightgbm as lgb

# Data Processing Class
class AdvancedDataProcessor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        """Download stock data"""
        print(f"Downloading {self.ticker} data...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        print(f"Download complete! Total {len(self.data)} records")
        return self.data

    def add_advanced_features(self):
        """Add technical indicator features"""
        print("🔧 Calculating technical indicators...")
        df = self.data.copy()
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = SMAIndicator(close, window=period).sma_indicator()
            df[f'EMA_{period}'] = EMAIndicator(close, window=period).ema_indicator()

        # Moving average slopes
        df['SMA_20_slope'] = df['SMA_20'].diff(5) / df['SMA_20'].shift(5)
        df['SMA_50_slope'] = df['SMA_50'].diff(10) / df['SMA_50'].shift(10)

        # RSI multiple periods
        for period in [9, 14, 25]:
            df[f'RSI_{period}'] = RSIIndicator(close, window=period).rsi()

        # MACD
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # ATR
        atr = AverageTrueRange(high, low, close, window=14)
        df['ATR'] = atr.average_true_range()
        df['ATR_Percent'] = df['ATR'] / close * 100

        # Bollinger Bands
        bb = BollingerBands(close)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / bb.bollinger_mavg()

        # OBV
        df['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # Price rate of change
        df['Price_ROC_5'] = close.pct_change(5)
        df['Price_ROC_10'] = close.pct_change(10)

        # ICT Concepts
        print("Calculating ICT indicators...")

        # 1. Fair Value Gaps (FVG) Detection
        df['FVG_Bullish'] = self._detect_bullish_fvg(df)
        df['FVG_Bearish'] = self._detect_bearish_fvg(df)
        df['FVG_Score'] = df['FVG_Bullish'] - df['FVG_Bearish']

        # 2. Order Blocks (Bullish & Bearish)
        df['Bullish_OB'] = self._detect_bullish_order_block(df)
        df['Bearish_OB'] = self._detect_bearish_order_block(df)
        df['OB_Score'] = df['Bullish_OB'] - df['Bearish_OB']

        # 3. Liquidity Sweeps (Stop Hunts)
        df['Liquidity_Sweep_High'] = self._detect_liquidity_sweep_high(df)
        df['Liquidity_Sweep_Low'] = self._detect_liquidity_sweep_low(df)
        df['Liquidity_Score'] = df['Liquidity_Sweep_Low'] - df['Liquidity_Sweep_High']

        # 4. Market Structure (Higher Highs / Lower Lows)
        df['Market_Structure'] = self._calculate_market_structure(df)
        df['Structure_Break'] = self._detect_structure_break(df)

        # 5. Balanced Price Range (Premium/Discount)
        df['Premium_Zone'] = self._calculate_premium_discount(df, zone_type='premium')
        df['Discount_Zone'] = self._calculate_premium_discount(df, zone_type='discount')

        # 6. Session High/Low (Asian, London, NY)
        df = self._calculate_session_extremes(df)

        # 7. Displacement (Strong momentum candles)
        df['Displacement'] = self._detect_displacement(df)

        # 8. Breaker Blocks (Failed Order Blocks)
        df['Breaker_Block'] = self._detect_breaker_blocks(df)

        print("ICT indicators calculated!")

        # Add VIX for market sentiment
        vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)['Close']
        vix_data = vix_data.reindex(df.index).ffill()
        df['VIX'] = vix_data

        # Target variable (5-day future return)
        df['Target_Return_5d'] = close.shift(-5) / close - 1
        # Reverted to binary target for consistency with backtesting and predict_ensemble
        # 1 for significant upward move, 0 otherwise
        threshold = 0.005 # Define a threshold for a significant upward move, e.g., 0.5%
        df['Target_Direction'] = np.where(df['Target_Return_5d'] > threshold, 1, 0)

        self.data = df.dropna()
        print(f"Feature engineering complete! Total {len(df.columns)} columns")
        return self.data

    # ICT Detection Methods
    def _detect_bullish_fvg(self, df):
        """Detect Bullish Fair Value Gap (3-candle gap up)"""
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        # Bullish FVG: Low[i] > High[i-2]
        fvg = (low > high.shift(2)).astype(int)
        return fvg

    def _detect_bearish_fvg(self, df):
        """Detect Bearish Fair Value Gap (3-candle gap down)"""
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        # Bearish FVG: High[i] < Low[i-2]
        fvg = (high < low.shift(2)).astype(int)
        return fvg

    def _detect_bullish_order_block(self, df):
        """Detect Bullish Order Block (Last down candle before strong up move)"""
        open_price = df['Open'].squeeze()
        close = df['Close'].squeeze()

        down_candle = (close < open_price).astype(int)
        next_up_candle = (close.shift(-1) > open_price.shift(-1)).astype(int)
        strong_move = ((close.shift(-1) - open_price.shift(-1)) / open_price.shift(-1) > 0.01).astype(int)

        bullish_ob = down_candle * next_up_candle * strong_move
        return bullish_ob

    def _detect_bearish_order_block(self, df):
        """Detect Bearish Order Block (Last up candle before strong down move)"""
        open_price = df['Open'].squeeze()
        close = df['Close'].squeeze()

        up_candle = (close > open_price).astype(int)
        next_down_candle = (close.shift(-1) < open_price.shift(-1)).astype(int)
        strong_move = ((open_price.shift(-1) - close.shift(-1)) / open_price.shift(-1) > 0.01).astype(int)

        bearish_ob = up_candle * next_down_candle * strong_move
        return bearish_ob

    def _detect_liquidity_sweep_high(self, df):
        """Detect Liquidity Sweep above recent highs (Stop Hunt)"""
        high = df['High'].squeeze()
        close = df['Close'].squeeze()

        # Find recent swing high (20-period high)
        swing_high = high.rolling(window=20).max()

        # Price breaks above then closes below
        sweep = ((high > swing_high.shift(1)) & (close < swing_high.shift(1))).astype(int)
        return sweep

    def _detect_liquidity_sweep_low(self, df):
        """Detect Liquidity Sweep below recent lows (Stop Hunt)"""
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()

        # Find recent swing low (20-period low)
        swing_low = low.rolling(window=20).min()

        # Price breaks below then closes above
        sweep = ((low < swing_low.shift(1)) & (close > swing_low.shift(1))).astype(int)
        return sweep

    def _calculate_market_structure(self, df):
        """Calculate Market Structure (1: Bullish, -1: Bearish, 0: Neutral)"""
        close = df['Close'].squeeze()

        # Higher Highs and Higher Lows = Bullish
        hh = (close > close.shift(10)).astype(int)
        hl = (close.rolling(10).min() > close.shift(20).rolling(10).min()).astype(int)

        # Lower Highs and Lower Lows = Bearish
        lh = (close < close.shift(10)).astype(int)
        ll = (close.rolling(10).min() < close.shift(20).rolling(10).min()).astype(int)

        structure = (hh * hl) - (lh * ll)
        return structure

    def _detect_structure_break(self, df):
        """Detect Break of Structure (BOS)"""
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()

        # Recent swing high/low (20 periods)
        swing_high = high.rolling(20).max()
        swing_low = low.rolling(20).min()

        # Bullish BOS: Close above recent swing high
        bullish_bos = (close > swing_high.shift(1)).astype(int)

        # Bearish BOS: Close below recent swing low
        bearish_bos = (close < swing_low.shift(1)).astype(int)

        structure_break = bullish_bos - bearish_bos
        return structure_break

    def _calculate_premium_discount(self, df, zone_type='premium'):
        """Calculate Premium/Discount zones (based on recent range)"""
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()

        # Calculate 50-period high/low
        period_high = high.rolling(50).max()
        period_low = low.rolling(50).min()
        range_50 = period_high - period_low

        # 50% equilibrium
        equilibrium = (period_high + period_low) / 2

        if zone_type == 'premium':
            # Premium zone: top 25% of range
            premium_threshold = equilibrium + (range_50 * 0.25)
            in_premium = (close > premium_threshold).astype(int)
            return in_premium
        else: # discount
            # Discount zone: bottom 25% of range
            discount_threshold = equilibrium - (range_50 * 0.25)
            in_discount = (close < discount_threshold).astype(int)
            return in_discount

    def _calculate_session_extremes(self, df):
        """Calculate session highs/lows (simplified - not time-based)"""
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()

        # Using rolling windows as proxy for sessions
        df['Asian_High'] = high.rolling(20).max()
        df['Asian_Low'] = low.rolling(20).min()
        df['London_High'] = high.rolling(30).max()
        df['London_Low'] = low.rolling(30).min()

        # Distance from session extremes
        df['Distance_From_High'] = (df['London_High'] - close) / close
        df['Distance_From_Low'] = (close - df['London_Low']) / close

        return df

    def _detect_displacement(self, df):
        """Detect Displacement (Strong directional candles)"""
        open_price = df['Open'].squeeze()
        close = df['Close'].squeeze()
        atr = df['ATR'].squeeze()

        # Candle size relative to ATR
        candle_size = abs(close - open_price)
        displacement = (candle_size > atr * 1.5).astype(int)

        # Direction of displacement
        displacement_direction = np.where(close > open_price, displacement, -displacement)
        return pd.Series(displacement_direction, index=df.index)

    def _detect_breaker_blocks(self, df):
        """Detect Breaker Blocks (Failed Order Blocks that reverse)"""
        close = df['Close'].squeeze()

        # Simplified: Look for failed support/resistance
        support_break = (close < close.shift(5).rolling(10).min()).astype(int)
        resistance_break = (close > close.shift(5).rolling(10).max()).astype(int)

        breaker = resistance_break - support_break
        return breaker
# LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1) # sigmoid

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1]) # 取最後一層 hidden state
        return torch.sigmoid(out) # 輸出 0~1 機率
# 序列資料集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )
# LSTM Training Function
def train_lstm(X_scaled, y, seq_len=30, epochs=100, batch_size=64, device='cpu'):
    dataset = TimeSeriesDataset(X_scaled, y, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = StockLSTM(
        input_size=X_scaled.shape[1],
        hidden_size=64,
        num_layers=2,
        dropout=0.25
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device).unsqueeze(1)

            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} loss: {total_loss/len(loader):.5f}")

    return model
# LSTM Predict Function
def predict_lstm(model, X_scaled, seq_len=30, device='cpu'):
    model.eval()
    probs = []

    with torch.no_grad():
        for i in range(seq_len, len(X_scaled)):
            seq = torch.tensor(X_scaled[i-seq_len:i], dtype=torch.float32).unsqueeze(0).to(device)
            prob = model(seq).cpu().numpy()[0,0]
            probs.append(prob)

    # 前 seq_len 筆因為沒有完整序列，補 0.5
    return np.array([0.5]*seq_len + probs)
# Ensemble Learning Model
class EnsemblePredictor:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.scaler = RobustScaler()
        self.models = {}
        self.ensemble_weights = {}
        self.lstm_model = None
        self.seq_len = 30 # LSTM seq_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用裝置：{self.device}")

    def create_models(self):
        """Create three models"""
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1,
            ),
            'xgb': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                objective='binary:logistic', # Changed to binary objective
                verbosity=0
            ),
            'lgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                random_state=42,
                objective='binary', # Changed to binary objective
                verbose=-1
            )
        }
        return self.models

    def prepare_data(self, data, target_col='Target_Direction'):
        """Data standardization"""
        X = data[self.feature_columns].values
        y = data[target_col].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train_lstm_model(self, X_scaled, y):
        print("Start training LSTM...")
        self.lstm_model = train_lstm(
            X_scaled, y,
            seq_len=self.seq_len,
            epochs=60,
            batch_size=64,
            device=self.device
        )
        print("LSTM trained！")

    def predict_lstm_proba(self, X_scaled):
        if self.lstm_model is None:
            raise ValueError("請先訓練 LSTM 模型")
        return predict_lstm(self.lstm_model, X_scaled, self.seq_len, self.device)

    def train_ensemble(self, X, y):
        """Train ensemble model"""
        print("Starting model training...")
        self.create_models()
        tscv = TimeSeriesSplit(n_splits=3)

        for name, model in self.models.items():
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                cv_scores.append(score)

            model.fit(X, y)
            avg_score = np.mean(cv_scores)
            self.ensemble_weights[name] = avg_score
            print(f" {name.upper()}: Accuracy {avg_score:.2%}")

        self.train_lstm_model(X, y)

        # 取得 LSTM 在訓練集上的 oof 機率（近似）
        lstm_oof_proba = self.predict_lstm_proba(X)
        # 簡單用準確率當權重
        lstm_accuracy = ((lstm_oof_proba > 0.5) == y).mean()
        self.ensemble_weights['lstm'] = lstm_accuracy

        # 重新正規化權重
        total = sum(self.ensemble_weights.values())
        self.ensemble_weights = {k: v/total for k, v in self.ensemble_weights.items()}
        print(f"LSTM CV-like Accuracy: {lstm_accuracy:.2%}")
        print("All Model Weights：", self.ensemble_weights)
        print("Model training complete!")
        return self.models

    def predict_ensemble(self, X):
        """Ensemble prediction"""
        predictions = []
        for name, model in self.models.items():
            # For binary classification models, predict_proba returns a 2-column array (proba_class_0, proba_class_1)
            # We want the probability of class 1 (upward movement)
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba * self.ensemble_weights[name])

        # LSTM predict
        lstm_proba = self.predict_lstm_proba(X)
        predictions.append(lstm_proba * self.ensemble_weights.get('lstm', 0.0))

        ensemble_proba = np.sum(predictions, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        return ensemble_pred, ensemble_proba

    def save_models(self, save_dir: str):
        """儲存所有模型與相關物件"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        metadata = {
            "ticker": self.ticker if hasattr(self, 'ticker') else "unknown",
            "version": "v20260112",
            "feature_columns": self.feature_columns,
            "seq_len": self.seq_len,
            "ensemble_weights": self.ensemble_weights,
            "scaler_params": {
                "center": self.scaler.center_.tolist() if hasattr(self.scaler, 'center_') else [],
                "scale": self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else []
            },
            "timestamp": datetime.now().isoformat()
        }

        # 1. 儲存樹模型
        for name, model in self.models.items():
            # Use joblib.dump for all scikit-learn compatible models
            joblib.dump(model, f"{save_dir}/{name}_model.joblib")

        # 2. 儲存 LSTM
        if self.lstm_model is not None:
            torch.save(self.lstm_model.state_dict(), f"{save_dir}/lstm_state_dict.pth")

        # 3. 儲存 metadata
        with open(f"{save_dir}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"模型儲存完成於：{save_dir}")
        print("包含檔案：")
        for f in sorted(os.listdir(save_dir)):
            print(f"  • {f}")

    def load_models(self, load_dir: str):
        """從資料夾載入所有模型與相關物件"""
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"模型資料夾不存在：{load_dir}")

        # 讀取 metadata
        with open(f"{load_dir}/metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.feature_columns = metadata["feature_columns"]
        self.seq_len = metadata["seq_len"]
        self.ensemble_weights = metadata["ensemble_weights"]

        # 還原 scaler
        self.scaler = RobustScaler()
        if metadata["scaler_params"]["center"]:
            self.scaler.center_ = np.array(metadata["scaler_params"]["center"])
        if metadata["scaler_params"]["scale"]:
            self.scaler.scale_ = np.array(metadata["scaler_params"]["scale"])

        # 載入樹模型
        self.models = {}
        for name in ['rf', 'xgb', 'lgbm']:
            # Use joblib.load for all scikit-learn compatible models
            self.models[name] = joblib.load(f"{load_dir}/{name}_model.joblib")

        # 載入 LSTM
        input_size = len(self.feature_columns)
        self.lstm_model = StockLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.25
        )
        state_dict = torch.load(f"{load_dir}/lstm_state_dict.pth",
                               map_location=self.device)
        self.lstm_model.load_state_dict(state_dict)
        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        print(f"成功載入模型版本：{metadata.get('version', '未知')}")
        print(f"特徵數量：{len(self.feature_columns)}")
        return True

# Risk Management
class RiskManager:
    def __init__(self, initial_capital, max_risk_per_trade=0.02):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_position_size(self, price, confidence, atr):
        """Calculate position size"""
        # Kelly Criterion
        win_rate = confidence
        kelly_fraction = max(0, min((win_rate * 0.015 - (1 - win_rate) * 0.01) / 0.015, 0.25))

        # ATR risk control
        risk_amount = self.capital * self.max_risk_per_trade
        shares_by_risk = int(risk_amount / (atr * 2.0))
        shares_by_kelly = int(self.capital * kelly_fraction / price)

        shares = min(shares_by_risk, shares_by_kelly)
        return max(0, min(shares, int(self.capital * 0.95 / price)))

    def calculate_stop_loss(self, entry_price, atr):
        """Calculate stop loss price"""
        return entry_price - (atr * 2.0)

    def calculate_take_profit(self, entry_price, stop_loss):
        """Calculate take profit price (2:1 risk-reward ratio)"""
        risk = abs(entry_price - stop_loss)
        return entry_price + (risk * 2.0)
# Backtesting System
class AdvancedBacktester:
    def __init__(self, initial_capital=100000):
        self.risk_manager = RiskManager(initial_capital)
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def run_backtest(self, data, predictions, probabilities, atr_values):
        """Execute backtest"""
        print("Running backtest...")

        for i in range(len(data)):
            date = data.index[i]
            price = float(data['Close'].iloc[i])
            signal = predictions[i]
            confidence = probabilities[i]
            atr = atr_values[i]

            # Calculate current equity
            current_equity = self.risk_manager.capital
            for pos in self.positions.values():
                current_equity += pos['shares'] * price

            self.equity_curve.append({'date': date, 'equity': current_equity})

            # Check stop loss/take profit
            for ticker in list(self.positions.keys()):
                pos = self.positions[ticker]
                if price <= pos['stop_loss'] or price >= pos['take_profit']:
                    revenue = pos['shares'] * price * 0.999 # 0.1% commission
                    self.risk_manager.capital += revenue
                    pnl = revenue - (pos['shares'] * pos['entry_price'])

                    exit_reason = 'STOP_LOSS' if price <= pos['stop_loss'] else 'TAKE_PROFIT'
                    self.trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'reason': exit_reason
                    })
                    del self.positions[ticker]

            # Entry logic
            if signal == 1 and len(self.positions) == 0 and confidence > 0.6:
                shares = self.risk_manager.calculate_position_size(price, confidence, atr)
                if shares > 0:
                    cost = shares * price * 1.001 # 0.1% commission
                    if cost <= self.risk_manager.capital:
                        stop_loss = self.risk_manager.calculate_stop_loss(price, atr)
                        take_profit = self.risk_manager.calculate_take_profit(price, stop_loss)

                        self.positions[i] = {
                            'shares': shares,
                            'entry_price': price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        self.risk_manager.capital -= cost
                        self.trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': price,
                            'shares': shares
                        })

        print("Backtest complete!")
        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate performance metrics"""
        equity_df = pd.DataFrame(self.equity_curve)

        # Total return
        total_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]

        # Sharpe Ratio
        returns = equity_df['equity'].pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        win_rate = len([t for t in sell_trades if t['pnl'] > 0]) / len(sell_trades) if sell_trades else 0

        # Average win/loss
        winning_trades = [t['pnl'] for t in sell_trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in sell_trades if t['pnl'] < 0]
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(sell_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': equity_df
        }
# Visualization Functions
def plot_data_overview(data, ticker):
    """1. Data Overview"""
    print("\n" + "="*60)
    print("Step 1: Data Overview")
    print("="*60)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{ticker} Data Overview & Basic Statistics', fontsize=16, fontweight='bold')
    # 1. Price trend
    axes[0, 0].plot(data.index, data['Close'], linewidth=1.5, color='#2563eb')
    axes[0, 0].fill_between(data.index, data['Low'].squeeze(), data['High'].squeeze(), alpha=0.2, color='#2563eb')
    axes[0, 0].set_title('Price Trend (with High-Low Range)', fontweight='bold')
    axes[0, 0].set_ylabel('Price (USD)')
    axes[0, 0].grid(True, alpha=0.3)
    # 2. Volume
    axes[0, 1].bar(data.index, data['Volume'].squeeze(), width=1, color='#10b981', alpha=0.6)
    axes[0, 1].set_title('Volume Distribution', fontweight='bold')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True, alpha=0.3)
    # 3. Daily returns
    returns = data['Close'].pct_change().dropna()
    axes[1, 0].hist(returns, bins=50, color='#8b5cf6', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(returns.mean().item(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean().item():.4f}')
    axes[1, 0].set_title('Daily Returns Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Returns')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # 4. Statistics table
    stats_text = f"""
    DATA STATISTICS SUMMARY
    ═══════════════════════════════

    Period : {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}
    Trading Days : {len(data):>8d} days

    Price Statistics:
    ───────────────────────────────
    Highest : ${data['High'].max().item():>10.2f}
    Lowest : ${data['Low'].min().item():>10.2f}
    Average : ${data['Close'].mean().item():>10.2f}
    Std Dev : ${data['Close'].std().item():>10.2f}

    Return Statistics:
    ───────────────────────────────
    Avg Daily Return: {returns.mean().item():>10.4f}
    Volatility : {returns.std().item():>10.4f}
    Max Daily Gain : {returns.max().item():>10.2%}
    Max Daily Loss : {returns.min().item():>10.2%}
    """
    axes[1, 1].text(0.08, 0.5, stats_text,
                    fontsize=10,
                    fontfamily='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.7',
                              facecolor='#f0f9ff',
                              edgecolor='#2563eb',
                              linewidth=1.5,
                              alpha=0.95))

    axes[1, 1].axis('off')
    axes[1, 1].set_title('Data Statistics Summary', fontsize=13, fontweight='bold', pad=12)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Data Statistics Summary', fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("Data overview complete!\n")
def plot_feature_importance(predictor, feature_cols):
    """2. Feature Importance Analysis"""
    print("="*60)
    print("Step 2: Feature Importance Analysis")
    print("="*60)
    # Get Random Forest feature importance
    rf_importances = predictor.models['rf'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_importances
    }).sort_values('importance', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color='#2563eb')
    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels(feature_importance_df['feature'])
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title('Top 20 Important Features (Random Forest)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                ha='left', va='center', fontsize=9, color='black')
    plt.tight_layout()
    plt.show()
    print("Feature importance analysis complete!\n")
def plot_model_performance(predictor, X_train, y_train, X_test, y_test):
    print("="*60)
    print("Step 3: Model Training Results")
    print("="*60)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Model Training Results Analysis', fontsize=16, fontweight='bold')

    # 1. Model accuracy comparison （只顯示樹模型，因為 LSTM 沒有 .score()）
    train_scores = []
    test_scores = []
    model_names = []
    for name, model in predictor.models.items():
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        model_names.append(name.upper())

    x = np.arange(len(model_names))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_scores, width, label='Train', color='#10b981', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_scores, width, label='Test', color='#2563eb', alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Tree Models Accuracy Comparison', fontweight='bold') # 改標題更清楚
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0.4, 1.0])

    # 2. Ensemble weights
    weights = list(predictor.ensemble_weights.values())
    all_model_names = list(predictor.ensemble_weights.keys()) # ['rf','xgb','lgbm','lstm']
    all_model_names = [name.upper() for name in all_model_names] # 轉大寫

    colors = ['#10b981', '#2563eb', '#8b5cf6', '#f59e0b']
    axes[0, 1].pie(weights, labels=all_model_names, autopct='%1.1f%%',
                   startangle=90, colors=colors, textprops={'fontsize': 10})
    axes[0, 1].set_title('Ensemble Model Weight Distribution\n(including LSTM)',
                         fontweight='bold')

    # 3. Prediction probability distribution
    _, test_proba = predictor.predict_ensemble(X_test)
    axes[1, 0].hist(test_proba, bins=30, color='#8b5cf6', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    axes[1, 0].set_xlabel('Prediction Probability (Bullish)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Ensemble Prediction Probability Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Confusion matrix
    from sklearn.metrics import confusion_matrix
    test_pred, _ = predictor.predict_ensemble(X_test)
    cm = confusion_matrix(y_test, test_pred)
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 1].set_title('Confusion Matrix (Test Set)', fontweight='bold')
    tick_marks = np.arange(2)
    axes[1, 1].set_xticks(tick_marks)
    axes[1, 1].set_yticks(tick_marks)
    axes[1, 1].set_xticklabels(['Down', 'Up'])
    axes[1, 1].set_yticklabels(['Down', 'Up'])
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_xlabel('Predicted')
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center',
                           color='white' if cm[i, j] > cm.max()/2 else 'black',
                           fontsize=20, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 1])

    # Add precision and recall
    test_pred, _ = predictor.predict_ensemble(X_test)
    precision = precision_score(y_test, test_pred, average='weighted')
    recall = recall_score(y_test, test_pred, average='weighted')
    print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")

    plt.tight_layout()
    plt.show()
    print("Model training results analysis complete!\n")
def plot_technical_analysis(data_test):
    """4. Technical Analysis Charts (with ICT)"""
    print("="*60)
    print("Step 4: Complete Technical Analysis Charts (ICT Enhanced)")
    print("="*60)
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)
    # 1. Price with ICT zones
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data_test.index, data_test['Close'], label='Close', linewidth=2, color='black')
    ax1.plot(data_test.index, data_test['SMA_20'], label='SMA 20', linewidth=1.5, color='#2563eb', alpha=0.8)
    ax1.plot(data_test.index, data_test['SMA_50'], label='SMA 50', linewidth=1.5, color='#10b981', alpha=0.8)
    # Mark Order Blocks
    bullish_ob_dates = data_test[data_test['Bullish_OB'] == 1].index
    bearish_ob_dates = data_test[data_test['Bearish_OB'] == 1].index
    ax1.scatter(bullish_ob_dates, data_test.loc[bullish_ob_dates, 'Low'],
                color='green', marker='^', s=100, alpha=0.7, label='Bullish OB', zorder=5)
    ax1.scatter(bearish_ob_dates, data_test.loc[bearish_ob_dates, 'High'],
                color='red', marker='v', s=100, alpha=0.7, label='Bearish OB', zorder=5)
    # Mark FVG zones
    fvg_bull_dates = data_test[data_test['FVG_Bullish'] == 1].index
    fvg_bear_dates = data_test[data_test['FVG_Bearish'] == 1].index
    ax1.scatter(fvg_bull_dates, data_test.loc[fvg_bull_dates, 'Low'],
                color='cyan', marker='D', s=60, alpha=0.6, label='Bullish FVG', zorder=4)
    ax1.scatter(fvg_bear_dates, data_test.loc[fvg_bear_dates, 'High'],
                color='magenta', marker='D', s=60, alpha=0.6, label='Bearish FVG', zorder=4)
    ax1.set_title('Price with ICT Concepts (Order Blocks & FVG)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left', ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)
    # 2. Market Structure
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data_test.index, data_test['Market_Structure'], linewidth=1.5, color='#2563eb', label='Market Structure')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.fill_between(data_test.index, 0, data_test['Market_Structure'],
                     where=(data_test['Market_Structure'] > 0), alpha=0.3, color='green', label='Bullish')
    ax2.fill_between(data_test.index, 0, data_test['Market_Structure'],
                     where=(data_test['Market_Structure'] < 0), alpha=0.3, color='red', label='Bearish')
    ax2.set_title('ICT Market Structure', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Structure Score')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    # 3. Order Block Score
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data_test.index, data_test['OB_Score'], linewidth=1.5, color='#8b5cf6')
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.fill_between(data_test.index, 0, data_test['OB_Score'], alpha=0.3, color='purple')
    ax3.set_title('ICT Order Block Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('OB Score')
    ax3.grid(True, alpha=0.3)
    # 4. Fair Value Gap Score
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(data_test.index, data_test['FVG_Score'], linewidth=1.5, color='#f59e0b')
    ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax4.fill_between(data_test.index, 0, data_test['FVG_Score'], alpha=0.3, color='orange')
    ax4.set_title('ICT Fair Value Gap Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('FVG Score')
    ax4.grid(True, alpha=0.3)
    # 5. Liquidity Sweeps
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(data_test.index, data_test['Liquidity_Score'], linewidth=1.5, color='#10b981')
    ax5.axhline(0, color='black', linewidth=0.8, linestyle='--')
    sweep_high_dates = data_test[data_test['Liquidity_Sweep_High'] == 1].index
    sweep_low_dates = data_test[data_test['Liquidity_Sweep_Low'] == 1].index
    ax5.scatter(sweep_high_dates, [0.5] * len(sweep_high_dates),
                color='red', marker='v', s=100, alpha=0.7, label='High Sweep')
    ax5.scatter(sweep_low_dates, [-0.5] * len(sweep_low_dates),
                color='green', marker='^', s=100, alpha=0.7, label='Low Sweep')
    ax5.set_title('ICT Liquidity Sweeps (Stop Hunts)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Liquidity Score')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)
    # 6. Premium/Discount Zones
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(data_test.index, data_test['Close'], linewidth=1.5, color='black', alpha=0.5)
    premium_dates = data_test[data_test['Premium_Zone'] == 1].index
    discount_dates = data_test[data_test['Discount_Zone'] == 1].index
    ax6.scatter(premium_dates, data_test.loc[premium_dates, 'Close'],
                color='red', alpha=0.3, s=30, label='Premium Zone')
    ax6.scatter(discount_dates, data_test.loc[discount_dates, 'Close'],
                color='green', alpha=0.3, s=30, label='Discount Zone')
    ax6.set_title('ICT Premium/Discount Zones', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Price (USD)')
    ax6.legend(loc='upper left', fontsize=9)
    ax6.grid(True, alpha=0.3)
    # 7. Displacement
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(data_test.index, data_test['Displacement'], linewidth=1.5, color='#ef4444')
    ax7.axhline(0, color='black', linewidth=0.8)
    ax7.fill_between(data_test.index, 0, data_test['Displacement'],
                     where=(data_test['Displacement'] > 0), alpha=0.3, color='green')
    ax7.fill_between(data_test.index, 0, data_test['Displacement'],
                     where=(data_test['Displacement'] < 0), alpha=0.3, color='red')
    ax7.set_title('ICT Displacement (Strong Moves)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Displacement')
    ax7.grid(True, alpha=0.3)
    # 8. RSI (Traditional)
    ax8 = fig.add_subplot(gs[4, 0])
    ax8.plot(data_test.index, data_test['RSI_14'], label='RSI 14', linewidth=1.5, color='#8b5cf6')
    ax8.axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
    ax8.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax8.fill_between(data_test.index, 30, 70, alpha=0.1, color='gray')
    ax8.set_title('RSI - Relative Strength Index', fontsize=12, fontweight='bold')
    ax8.set_ylabel('RSI')
    ax8.set_ylim([0, 100])
    ax8.legend(loc='upper left', fontsize=9)
    ax8.grid(True, alpha=0.3)
    # 9. MACD (Traditional)
    ax9 = fig.add_subplot(gs[4, 1])
    ax9.plot(data_test.index, data_test['MACD'], label='MACD', linewidth=1.5, color='#2563eb')
    ax9.plot(data_test.index, data_test['MACD_Signal'], label='Signal', linewidth=1.5, color='#ef4444')
    ax9.bar(data_test.index, data_test['MACD_Hist'], label='Histogram', alpha=0.3, color='gray')
    ax9.axhline(0, color='black', linewidth=0.8)
    ax9.set_title('MACD Indicator', fontsize=12, fontweight='bold')
    ax9.set_ylabel('MACD')
    ax9.legend(loc='upper left', fontsize=9)
    ax9.grid(True, alpha=0.3)
    plt.suptitle('Complete Technical Analysis Dashboard (ICT Enhanced)', fontsize=16, fontweight='bold', y=0.996)
    plt.tight_layout()
    plt.show()
    print("Technical analysis charts complete!\n")
def plot_backtest_results(metrics, data_test, predictions, backtester):
    """5. Backtest Results Analysis - FIXED VERSION"""
    print("="*60)
    print("Step 5: Complete Backtest Results Analysis")
    print("="*60)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3) # FIXED: Increased hspace from 0.3 to 0.4
    equity_df = metrics['equity_curve']
    # 1. Equity curve and drawdown
    ax1 = fig.add_subplot(gs[0, :])
    ax1_twin = ax1.twinx()
    ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2.5, color='#2563eb', label='Equity Curve')
    ax1.axhline(metrics['equity_curve']['equity'].iloc[0], color='gray', linestyle='--', linewidth=1, label='Initial Capital')
    ax1.set_ylabel('Equity (USD)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=10)
    returns = equity_df['equity'].pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ax1_twin.fill_between(equity_df['date'].iloc[1:], 0, drawdown.values * 100,
                          alpha=0.3, color='red', label='Drawdown')
    ax1_twin.set_ylabel('Drawdown (%)', fontsize=10, fontweight='bold', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red', labelsize=9)
    ax1.set_title('Equity Curve & Drawdown Analysis', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1_twin.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)
    # 2. Price and trading signals
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data_test.index, data_test['Close'], label='Price', linewidth=1.5, color='black', alpha=0.7)
    for trade in backtester.trades:
        if trade['action'] == 'BUY':
            ax2.scatter(trade['date'], trade['price'], color='green', marker='^',
                       s=150, label='Buy' if 'BUY' not in [t.get_label() for t in ax2.collections] else '',
                       zorder=5, edgecolors='black', linewidth=1)
        elif trade['action'] == 'SELL':
            reason = trade.get('reason', '')
            color = 'red' if reason == 'STOP_LOSS' else 'blue'
            marker = 'v' if reason == 'STOP_LOSS' else 'D'
            label = 'Stop Loss' if reason == 'STOP_LOSS' and 'Stop Loss' not in [t.get_label() for t in ax2.collections] else \
                    'Take Profit' if reason == 'TAKE_PROFIT' and 'Take Profit' not in [t.get_label() for t in ax2.collections] else ''
            ax2.scatter(trade['date'], trade['price'], color=color, marker=marker,
                       s=150, label=label, zorder=5, edgecolors='black', linewidth=1)
    ax2.set_title('Trading Signals & Execution Points', fontsize=11, fontweight='bold', pad=8)
    ax2.set_ylabel('Price (USD)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    # 3. Trade P&L
    ax3 = fig.add_subplot(gs[1, 1])
    sell_trades = [t for t in backtester.trades if t['action'] == 'SELL']
    trade_pnls = [t['pnl'] for t in sell_trades]
    colors_pnl = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
    ax3.bar(range(len(trade_pnls)), trade_pnls, color=colors_pnl, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_title('Trade P&L Distribution', fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel('Trade Number', fontsize=10)
    ax3.set_ylabel('P&L (USD)', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=9)
    # 4. Performance metrics panel
    ax4 = fig.add_subplot(gs[2, 0])
    metrics_text = f"""
    Complete Performance Metrics
    ═══════════════════════════════

    Return & Risk:
    ───────────────────────────────
    Total Return : {metrics['total_return']:>10.2%}
    Sharpe Ratio : {metrics['sharpe_ratio']:>10.2f}
    Max Drawdown : {metrics['max_drawdown']:>10.2%}
    Trade Statistics:
    ───────────────────────────────
    Total Trades : {metrics['total_trades']:>10d}
    Winning Trades : {len([t for t in sell_trades if t['pnl'] > 0]):>10d}
    Losing Trades : {len([t for t in sell_trades if t['pnl'] < 0]):>10d}
    Win Rate : {metrics['win_rate']:>10.2%}
    Profitability:
    ───────────────────────────────
    Avg Win : ${metrics['avg_win']:>9.2f}
    Avg Loss : ${metrics['avg_loss']:>9.2f}
    Profit Factor : {abs(metrics['avg_win']/metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0:>10.2f}
    Capital Summary:
    ───────────────────────────────
    Initial Capital : ${equity_df['equity'].iloc[0]:>9.2f}
    Final Equity : ${equity_df['equity'].iloc[-1]:>9.2f}
    Net Profit : ${equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]:>9.2f}
    """
    ax4.text(0.08, 0.95, metrics_text,
            fontsize=9.5,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.7',
                      facecolor='#f8fafc',
                      edgecolor='#2563eb',
                      linewidth=1.5,
                      alpha=0.95))
    ax4.axis('off')
    ax4.set_title('Complete Performance Metrics', fontsize=12, fontweight='bold', pad=12)
    # 5. Returns distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(returns, bins=40, color='#8b5cf6', alpha=0.7, edgecolor='black')
    ax5.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {returns.mean():.4f}')
    ax5.axvline(returns.median(), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {returns.median():.4f}')
    ax5.set_title('Daily Returns Distribution', fontsize=11, fontweight='bold', pad=8)
    ax5.set_xlabel('Returns', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=9)
    # FIXED: Adjusted margins - more top margin
    plt.suptitle('Complete Backtest Results Analysis', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    plt.show()
    print("Backtest results analysis complete!\n")
# Future Prediction Class
class FuturePredictor:
    def __init__(self, predictor, processor):
        self.predictor = predictor
        self.processor = processor
    def predict_future_days(self, data, n_days=5):
        """Predict future N days based on latest data"""
        print(f"\n" + "="*60)
        print(f"🔮 Predicting Future {n_days} Days")
        print(f"="*60)
        # Use last 100 rows for prediction context
        latest_data = data.tail(100).copy()
        # Get the latest complete feature set
        X_latest = latest_data[self.predictor.feature_columns].values
        X_latest_scaled = self.predictor.scaler.transform(X_latest)
        # Predict on latest data
        predictions, probabilities = self.predictor.predict_ensemble(X_latest_scaled)
        # Get the most recent prediction
        latest_prediction = predictions[-1]
        latest_probability = probabilities[-1]
        latest_date = latest_data.index[-1]
        # Calculate future dates (skip weekends roughly)
        future_dates = []
        current_date = latest_date
        days_added = 0
        while days_added < n_days:
            current_date = current_date + pd.Timedelta(days=1)
            # Simple weekend skip (not perfect for holidays)
            if current_date.weekday() < 5: # Monday = 0, Friday = 4
                future_dates.append(current_date)
                days_added += 1
        # Get latest price and technical indicators
        latest_close = float(latest_data['Close'].iloc[-1])
        latest_atr = float(latest_data['ATR'].iloc[-1])
        latest_rsi = float(latest_data['RSI_14'].iloc[-1])
        latest_macd = float(latest_data['MACD'].iloc[-1])
        # Optimized: Estimate future price using ATR-based volatility
        estimated_changes = []
        daily_vol = latest_atr / latest_close  # ATR as % volatility
        expected_vol = daily_vol * 1.5  # Slightly amplify for projection
        strength = abs(latest_probability - 0.5) * 2  # 0~1 strength
        direction = 1 if latest_probability > 0.5 else -1
        for i in range(n_days):
            change_pct = direction * strength * expected_vol * 2.2  # Adjustable factor
            estimated_changes.append(change_pct)
        # Calculate projected prices
        projected_prices = [latest_close]
        for change in estimated_changes:
            new_price = projected_prices[-1] * (1 + change)
            projected_prices.append(new_price)
        # Create results dataframe
        results = pd.DataFrame({
            'Date': [latest_date] + future_dates,
            'Predicted_Price': projected_prices,
            'Change_Pct': [0] + estimated_changes,
            'Confidence': [latest_probability] * (n_days + 1)
        })
        # Display prediction summary
        print(f"\nLatest Market Data ({latest_date.strftime('%Y-%m-%d')}):")
        print(f" Current Price: ${latest_close:.2f}")
        print(f" ATR (Volatility): ${latest_atr:.2f}")
        print(f" RSI: {latest_rsi:.2f}")
        print(f" MACD: {latest_macd:.2f}")
        print(f"\nModel Prediction:")
        print(f" Direction: {'📈 BULLISH' if latest_probability > 0.5 else '📉 BEARISH'}")
        print(f" Confidence: {latest_probability:.1%}")
        print(f" Signal Strength: {'🟢 Strong' if abs(latest_probability - 0.5) > 0.15 else '🟡 Moderate' if abs(latest_probability - 0.5) > 0.08 else '🔴 Weak'}")
        print(f"\nFuture Price Projections:")
        print(f"{'='*60}")
        for i in range(1, len(results)):
            row = results.iloc[i]
            change_str = f"+{row['Change_Pct']:.2%}" if row['Change_Pct'] > 0 else f"{row['Change_Pct']:.2%}"
            arrow = "↗️" if row['Change_Pct'] > 0 else "↘️"
            print(f" {row['Date'].strftime('%Y-%m-%d')} {arrow} ${row['Predicted_Price']:.2f} ({change_str})")
        print(f"{'='*60}")
        total_change = (results['Predicted_Price'].iloc[-1] - results['Predicted_Price'].iloc[0]) / results['Predicted_Price'].iloc[0]
        print(f" {n_days}-Day Total Expected Change: {total_change:+.2%}")
        print(f"{'='*60}\n")
        return results, latest_data
    def plot_future_prediction(self, historical_data, prediction_results, ticker):
        """Visualize future predictions - FIXED VERSION"""
        print("Generating prediction visualization...\n")
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        # FIXED: Reduced fontsize and added more top margin
        fig.suptitle(f'{ticker} - Future Price Prediction Analysis', fontsize=14, fontweight='bold', y=0.98)
        # 1. Historical + Predicted Price
        ax1 = axes[0, 0]
        recent_data = historical_data.tail(60)
        ax1.plot(recent_data.index, recent_data['Close'],
                label='Historical Price', linewidth=2, color='#2563eb')
        ax1.plot(prediction_results['Date'], prediction_results['Predicted_Price'],
                label='Predicted Price', linewidth=2.5, color='#ef4444', linestyle='--', marker='o')
        ax1.axvline(prediction_results['Date'].iloc[0], color='gray',
                  linestyle=':', linewidth=1.5, alpha=0.7, label='Prediction Start')
        ax1.fill_between(prediction_results['Date'].iloc[1:],
                        prediction_results['Predicted_Price'].iloc[1:] * 0.98,
                        prediction_results['Predicted_Price'].iloc[1:] * 1.02,
                        alpha=0.2, color='red', label='Uncertainty Range (±2%)')
        ax1.set_title('Price Forecast', fontsize=11, fontweight='bold', pad=8)
        ax1.set_ylabel('Price (USD)', fontsize=10)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        # 2. Confidence Meter
        ax2 = axes[0, 1]
        confidence = prediction_results['Confidence'].iloc[-1]
        direction = "BULLISH" if confidence > 0.5 else "BEARISH"
        color = '#10b981' if confidence > 0.5 else '#ef4444'
        categories = ['Strong\nBearish', 'Weak\nBearish', 'Neutral', 'Weak\nBullish', 'Strong\nBullish']
        colors_gauge = ['#dc2626', '#f97316', '#fbbf24', '#84cc16', '#10b981']
        ax2.barh(categories, [1]*5, color=colors_gauge, alpha=0.3, height=0.6)
        if confidence < 0.3:
            y_pos = 0
        elif confidence < 0.4:
            y_pos = 1
        elif confidence < 0.6:
            y_pos = 2
        elif confidence < 0.7:
            y_pos = 3
        else:
            y_pos = 4
        ax2.scatter([confidence], [y_pos], s=500, color=color, marker='>',
                  edgecolors='black', linewidth=2, zorder=5)
        ax2.set_xlim([0, 1])
        ax2.set_xlabel('Confidence Level', fontweight='bold', fontsize=10)
        ax2.set_title(f'Model Confidence: {confidence:.1%} ({direction})',
                    fontsize=11, fontweight='bold', color=color, pad=8)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.tick_params(labelsize=8)
        # 3. Expected Daily Changes
        ax3 = axes[1, 0]
        daily_changes = prediction_results['Change_Pct'].iloc[1:] * 100
        colors_bar = ['green' if x > 0 else 'red' for x in daily_changes]
        dates_str = [d.strftime('%m/%d') for d in prediction_results['Date'].iloc[1:]]
        bars = ax3.bar(dates_str, daily_changes, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.axhline(0, color='black', linewidth=1)
        ax3.set_title('Expected Daily Price Changes', fontsize=11, fontweight='bold', pad=8)
        ax3.set_ylabel('Change (%)', fontsize=10)
        ax3.set_xlabel('Date', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(labelsize=8)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')
        # 4. Key Technical Indicators (Current)
        ax4 = axes[1, 1]
        latest_rsi = float(historical_data['RSI_14'].iloc[-1])
        latest_macd = float(historical_data['MACD'].iloc[-1])
        latest_atr = float(historical_data['ATR'].iloc[-1])
        latest_ob_score = float(historical_data['OB_Score'].iloc[-1])
        latest_fvg_score = float(historical_data['FVG_Score'].iloc[-1])
        latest_structure = float(historical_data['Market_Structure'].iloc[-1])
        indicators_text = f"""
        CURRENT TECHNICAL INDICATORS
        ═══════════════════════════════
        Traditional Indicators:
        ───────────────────────────────
        RSI (14) : {latest_rsi:>7.2f}
        MACD : {latest_macd:>7.2f}
        ATR : ${latest_atr:>7.2f}
        ICT Indicators:
        ───────────────────────────────
        Order Block : {latest_ob_score:>7.2f}
        FVG Score : {latest_fvg_score:>7.2f}
        Structure : {latest_structure:>7.2f}
        Signal Interpretation:
        ───────────────────────────────
        RSI : {'Overbought' if latest_rsi > 70 else 'Oversold' if latest_rsi < 30 else 'Neutral':<10}
        MACD : {'Bullish' if latest_macd > 0 else 'Bearish':<10}
        ICT : {'Bullish' if latest_structure > 0 else 'Bearish' if latest_structure < 0 else 'Neutral':<10}
        """
        ax4.text(0.08, 0.5, indicators_text,
                fontsize=9.5,
                fontfamily='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.6',
                          facecolor='#f8fafc',
                          edgecolor='#64748b',
                          linewidth=1.2,
                          alpha=0.95))
        ax4.axis('off')
        ax4.set_title('Technical Analysis Snapshot', fontsize=12, fontweight='bold', pad=12)
        # FIXED: Increased bottom margin and top margin
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.show()
        print("Prediction visualization complete!\n")
# PDF Report Generator
class PDFReportGenerator:
    def __init__(self, ticker, filename="Trading_Analysis_Report.pdf"):
        self.ticker = ticker
        self.filename = filename
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        # Section style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#059669'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))

        # Add a specific bold style derived from 'Normal'
        self.styles.add(ParagraphStyle(
            name='BoldNormal',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold'
        ))
    def save_plot_to_buffer(self):
        """Save current matplotlib plot to buffer"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    def generate_report(self, metrics, prediction_results, data, predictor):
        """Generate complete PDF report"""
        print("\n" + "="*60)
        print("📄 Generating PDF Report...")
        print("="*60)
        doc = SimpleDocTemplate(self.filename, pagesize=A4)
        # ========== Cover Page ==========
        self._add_cover_page()
        # ========== Executive Summary ==========
        self._add_executive_summary(metrics, prediction_results)
        # ========== Market Overview ==========
        self._add_market_overview(data)
        # ========== Model Performance ==========
        self._add_model_performance(metrics)
        # ========== Future Predictions ==========
        self._add_future_predictions(prediction_results, data)
        # ========== Technical Analysis ==========
        self._add_technical_analysis(data)
        # ========== Risk Analysis ==========
        self._add_risk_analysis(metrics)
        # ========== Feature Importance ==========
        self._add_feature_importance(predictor)
        # ========== Disclaimer ==========
        self._add_disclaimer()
        # Build PDF
        doc.build(self.story)
        print(f"PDF Report generated: {self.filename}")
        print("="*60 + "\n")
    def _add_cover_page(self):
        """Add cover page"""
        self.story.append(Spacer(1, 2*inch))
        title = Paragraph(f"<b>QUANTITATIVE TRADING</b><br/><b>ANALYSIS REPORT</b>",
                         self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))
        subtitle = Paragraph(f"<b>{self.ticker}</b> - ICT Enhanced Strategy",
                            self.styles['CustomSubtitle'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.3*inch))
        date_text = Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
                             self.styles['Normal'])
        date_text.alignment = TA_CENTER
        self.story.append(date_text)
        self.story.append(Spacer(1, 1*inch))
        # Report highlights box
        highlights = f"""
        <para align=center>
        <b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br/>
        <b>REPORT HIGHLIGHTS</b><br/>
        <b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br/><br/>
        ✓ Machine Learning Ensemble Models<br/>
        ✓ ICT (Inner Circle Trader) Concepts<br/>
        ✓ Advanced Risk Management<br/>
        ✓ Future Price Predictions<br/>
        ✓ Comprehensive Technical Analysis<br/>
        </para>
        """
        self.story.append(Paragraph(highlights, self.styles['Normal']))
        self.story.append(PageBreak())
    def _add_executive_summary(self, metrics, prediction_results):
        """Add executive summary"""
        self.story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        # Key metrics table
        latest_price = prediction_results['Predicted_Price'].iloc[0]
        future_price = prediction_results['Predicted_Price'].iloc[-1]
        price_change = (future_price - latest_price) / latest_price
        confidence = prediction_results['Confidence'].iloc[-1]
        direction = "BULLISH 📈" if confidence > 0.5 else "BEARISH 📉"
        summary_data = [
            ['METRIC', 'VALUE'],
            ['Current Price', f'${latest_price:.2f}'],
            ['Predicted Future Price', f'${future_price:.2f}'],
            ['Expected Change', f'{price_change:+.2%}'],
            ['Model Confidence', f'{confidence:.1%}'],
            ['Market Outlook', direction],
            ['', ''],
            ['Historical Performance', ''],
            ['Total Return', f'{metrics["total_return"]:+.2%}'],
            ['Sharpe Ratio', f'{metrics["sharpe_ratio"]:.2f}'],
            ['Max Drawdown', f'{metrics["max_drawdown"]:.2%}'],
            ['Win Rate', f'{metrics["win_rate"]:.1%}'],
            ['Total Trades', str(metrics['total_trades'])],
        ]
        # Convert all string literals in summary_data to Paragraph objects
        summary_data_flowables = []
        for row in summary_data:
            flowable_row = []
            for item in row:
                flowable_row.append(Paragraph(str(item), self.styles['Normal']))
            summary_data_flowables.append(flowable_row)
        # Apply bold to headers and specific rows
        summary_data_flowables[0] = [Paragraph(str(item), self.styles['BoldNormal']) for item in summary_data[0]]
        summary_data_flowables[7] = [Paragraph(str(item), self.styles['BoldNormal']) for item in summary_data[7]]
        table = Table(summary_data_flowables, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 7), (-1, 7), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#e0e7ff')),
        ]))
        self.story.append(table)
        self.story.append(PageBreak())
    def _add_market_overview(self, data):
        """Add market overview section"""
        self.story.append(Paragraph("MARKET OVERVIEW", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        latest = data.iloc[-1]
        returns = data['Close'].pct_change().dropna()
        overview_text = f"""
        <b>Data Period:</b> {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}<br/>
        <b>Total Trading Days:</b> {len(data)}<br/><br/>
        <b>Price Statistics:</b><br/>
        • Current Price: ${float(latest['Close'].item()):.2f}<br/>
        • 52-Week High: ${float(data['High'].max().item()):.2f}<br/>
        • 52-Week Low: ${float(data['Low'].min().item()):.2f}<br/>
        • Average Price: ${float(data['Close'].mean().item()):.2f}<br/>
        • Standard Dev: ${float(data['Close'].std().item()):.2f}<br/><br/>
        <b>Volatility Metrics:</b><br/>
        • Daily Return Std: {float(returns.std().item()):.4f}<br/>
        • Annualized Volatility: {float(returns.std().item() * np.sqrt(252)):.2%}<br/>
        • Current ATR: ${float(latest['ATR'].item()):.2f}<br/><br/>
        <b>Momentum Indicators:</b><br/>
        • RSI (14): {float(latest['RSI_14'].item()):.2f}<br/>
        • MACD: {float(latest['MACD'].item()):.2f}<br/>
        • Market Structure: {float(latest['Market_Structure'].item()):.2f}<br/>
        """
        self.story.append(Paragraph(overview_text, self.styles['Normal']))
        self.story.append(PageBreak())
    def _add_model_performance(self, metrics):
        """Add model performance section"""
        self.story.append(Paragraph("MODEL PERFORMANCE ANALYSIS", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        # Performance metrics
        perf_text = f"""
        <b>Backtesting Results:</b><br/><br/>
        The ensemble model was tested on historical data with the following results:<br/><br/>
        <b>Return Metrics:</b><br/>
        • Total Return: {metrics['total_return']:+.2%}<br/>
        • Risk-Adjusted Return (Sharpe): {metrics['sharpe_ratio']:.2f}<br/>
        • Maximum Drawdown: {metrics['max_drawdown']:.2%}<br/><br/>
        <b>Trading Statistics:</b><br/>
        • Total Trades Executed: {metrics['total_trades']}<br/>
        • Win Rate: {metrics['win_rate']:.1%}<br/>
        • Average Winning Trade: ${metrics['avg_win']:.2f}<br/>
        • Average Losing Trade: ${metrics['avg_loss']:.2f}<br/>
        • Profit Factor: {abs(metrics['avg_win']/metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0:.2f}<br/><br/>
        <b>Model Interpretation:</b><br/>
        {'✓ Strong Performance - Model shows consistent profitability' if metrics['total_return'] > 0.1 else '⚠ Moderate Performance - Further optimization recommended' if metrics['total_return'] > 0 else '✗ Underperformance - Strategy revision needed'}<br/>
        {'✓ Excellent Risk Management' if metrics['win_rate'] > 0.6 else '⚠ Acceptable Win Rate' if metrics['win_rate'] > 0.5 else '✗ Low Win Rate'}<br/>
        {'✓ Strong Sharpe Ratio' if metrics['sharpe_ratio'] > 1.5 else '⚠ Moderate Sharpe Ratio' if metrics['sharpe_ratio'] > 0.8 else '✗ Low Risk-Adjusted Returns'}<br/>
        """
        self.story.append(Paragraph(perf_text, self.styles['Normal']))
        self.story.append(PageBreak())
    def _add_future_predictions(self, prediction_results, data):
        """Add future predictions section"""
        self.story.append(Paragraph("FUTURE PRICE PREDICTIONS", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        confidence = prediction_results['Confidence'].iloc[-1]
        direction = "BULLISH" if confidence > 0.5 else "BEARISH"
        pred_intro = f"""
        <b>Model Outlook: {direction}</b><br/>
        <b>Confidence Level: {confidence:.1%}</b><br/><br/>
        The ensemble model predicts the following price movements based on current market conditions,
        technical indicators, and ICT concepts:<br/><br/>
        """
        self.story.append(Paragraph(pred_intro, self.styles['Normal']))
        self.story.append(Spacer(1, 0.1*inch))
        # Prediction table
        pred_data = [['Date', 'Predicted Price', 'Change %', 'Confidence']]
        for i in range(len(prediction_results)):
            row = prediction_results.iloc[i]
            pred_data.append([
                row['Date'].strftime('%Y-%m-%d'),
                f"${row['Predicted_Price']:.2f}",
                f"{row['Change_Pct']:+.2%}" if i > 0 else "Current",
                f"{row['Confidence']:.1%}"
            ])
        # Convert all string literals in pred_data to Paragraph objects
        pred_data_flowables = []
        for row in pred_data:
            flowable_row = []
            for item in row:
                flowable_row.append(Paragraph(str(item), self.styles['Normal']))
            pred_data_flowables.append(flowable_row)

        # Apply bold to headers
        pred_data_flowables[0] = [Paragraph(str(item), self.styles['BoldNormal']) for item in pred_data[0]]
        pred_table = Table(pred_data_flowables, colWidths=[1.5*inch, 1.5*inch, 1.2*inch, 1.2*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        self.story.append(pred_table)
        self.story.append(Spacer(1, 0.2*inch))
        # Prediction interpretation
        total_change = (prediction_results['Predicted_Price'].iloc[-1] -
                       prediction_results['Predicted_Price'].iloc[0]) / prediction_results['Predicted_Price'].iloc[0]
        interp_text = f"""
        <b>Interpretation:</b><br/>
        • Expected total change over prediction period: {total_change:+.2%}<br/>
        • Signal strength: {'🟢 Strong' if abs(confidence - 0.5) > 0.15 else '🟡 Moderate' if abs(confidence - 0.5) > 0.08 else '🔴 Weak'}<br/>
        • Recommendation: {'Consider LONG positions' if confidence > 0.6 else 'Consider SHORT positions' if confidence < 0.4 else 'NEUTRAL - Wait for clearer signals'}<br/>
        """
        self.story.append(Paragraph(interp_text, self.styles['Normal']))
        self.story.append(PageBreak())
    def _add_technical_analysis(self, data):
        """Add technical analysis section"""
        self.story.append(Paragraph("TECHNICAL ANALYSIS (ICT Enhanced)", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        latest = data.iloc[-1]
        tech_text = f"""
        <b>Traditional Technical Indicators:</b><br/>
        • RSI (14): {latest['RSI_14'].item():.2f} - {'Overbought' if latest['RSI_14'].item() > 70 else 'Oversold' if latest['RSI_14'].item() < 30 else 'Neutral'}<br/>
        • MACD: {latest['MACD'].item():.2f} - {'Bullish' if latest['MACD'].item() > 0 else 'Bearish'}<br/>
        • ATR: ${latest['ATR'].item():.2f}<br/>
        • Bollinger Band Width: {latest['BB_Width'].item():.4f}<br/><br/>
        <b>ICT Concepts Analysis:</b><br/>
        • Order Block Score: {latest['OB_Score'].item():.2f}<br/>
        • Fair Value Gap Score: {latest['FVG_Score'].item():.2f}<br/>
        • Market Structure: {latest['Market_Structure'].item():.2f} - {'Bullish Structure' if latest['Market_Structure'].item() > 0 else 'Bearish Structure' if latest['Market_Structure'].item() < 0 else 'Neutral'}<br/>
        • Liquidity Score: {latest['Liquidity_Score'].item():.2f}<br/>
        • Displacement: {latest['Displacement'].item():.2f}<br/>
        • Premium Zone: {'Yes' if latest['Premium_Zone'].item() == 1 else 'No'}<br/>
        • Discount Zone: {'Yes' if latest['Discount_Zone'].item() == 1 else 'No'}<br/><br/>
        <b>Key Observations:</b><br/>
        • Recent Order Blocks detected: {int(data['Bullish_OB'].tail(20).sum())} Bullish, {int(data['Bearish_OB'].tail(20).sum())} Bearish (last 20 days)<br/>
        • Fair Value Gaps: {int(data['FVG_Bullish'].tail(20).sum())} Bullish, {int(data['FVG_Bearish'].tail(20).sum())} Bearish (last 20 days)<br/>
        • Liquidity Sweeps: {int(data['Liquidity_Sweep_Low'].tail(20).sum())} Low sweeps, {int(data['Liquidity_Sweep_High'].tail(20).sum())} High sweeps (last 20 days)<br/>
        """
        self.story.append(Paragraph(tech_text, self.styles['Normal']))
        self.story.append(PageBreak())
    def _add_risk_analysis(self, metrics):
        """Add risk analysis section"""
        self.story.append(Paragraph("RISK ANALYSIS", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        risk_text = f"""
        <b>Risk Metrics:</b><br/><br/>
        <b>Maximum Drawdown: {metrics['max_drawdown']:.2%}</b><br/>
        {'✓ Low risk - Drawdown under control' if abs(metrics['max_drawdown']) < 0.15 else '⚠ Moderate risk - Monitor closely' if abs(metrics['max_drawdown']) < 0.25 else '✗ High risk - Significant drawdown observed'}<br/><br/>
        <b>Sharpe Ratio: {metrics['sharpe_ratio']:.2f}</b><br/>
        {'✓ Excellent risk-adjusted returns' if metrics['sharpe_ratio'] > 1.5 else '⚠ Acceptable risk-adjusted returns' if metrics['sharpe_ratio'] > 0.8 else '✗ Poor risk-adjusted returns'}<br/><br/>
        <b>Win Rate: {metrics['win_rate']:.1%}</b><br/>
        {'✓ Strong win rate' if metrics['win_rate'] > 0.6 else '⚠ Moderate win rate' if metrics['win_rate'] > 0.5 else '✗ Low win rate - Strategy needs improvement'}<br/><br/>
        <b>Risk Management Strategy:</b><br/>
        • Stop Loss: 2x ATR from entry<br/>
        • Take Profit: 2:1 Risk-Reward Ratio<br/>
        • Position Sizing: Kelly Criterion + ATR-based<br/>
        • Max Risk Per Trade: 2% of capital<br/><br/>
        <b>Recommendations:</b><br/>
        • Always use stop-loss orders<br/>
        • Monitor position sizes based on volatility<br/>
        • Diversify across multiple assets<br/>
        • Regular portfolio rebalancing<br/>
        • Stay informed about market conditions<br/>
        """
        self.story.append(Paragraph(risk_text, self.styles['Normal']))
        self.story.append(PageBreak())
    def _add_feature_importance(self, predictor):
        """Add feature importance section"""
        self.story.append(Paragraph("MODEL FEATURES & IMPORTANCE", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        # Get top 15 features
        rf_importances = predictor.models['rf'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': predictor.feature_columns,
            'importance': rf_importances
        }).sort_values('importance', ascending=False).head(15)
        feat_intro = f"""
        <b>Top 15 Most Important Features:</b><br/><br/>
        The following features have the highest impact on model predictions (Random Forest analysis):<br/><br/>
        """
        self.story.append(Paragraph(feat_intro, self.styles['Normal']))
        self.story.append(Spacer(1, 0.1*inch))
        # Feature importance table
        feat_data = [['Rank', 'Feature', 'Importance Score']]
        for i, (idx, row) in enumerate(feature_importance_df.iterrows(), 1):
            feat_data.append([str(i), row['feature'], f"{row['importance']:.4f}"])
        # Convert all string literals in feat_data to Paragraph objects
        feat_data_flowables = []
        for row in feat_data:
            flowable_row = []
            for item in row:
                flowable_row.append(Paragraph(str(item), self.styles['Normal']))
            feat_data_flowables.append(flowable_row)

        # Apply bold to headers
        feat_data_flowables[0] = [Paragraph(str(item), self.styles['BoldNormal']) for item in feat_data[0]]
        feat_table = Table(feat_data_flowables, colWidths=[0.8*inch, 3*inch, 1.5*inch])
        feat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c3aed')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (2, 1), (2, -1), 'RIGHT'),
        ]))
        self.story.append(feat_table)
        self.story.append(PageBreak())
    def _add_disclaimer(self):
        """Add disclaimer section"""
        self.story.append(Paragraph("IMPORTANT DISCLAIMER", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        disclaimer_text = f"""
        <b>PLEASE READ CAREFULLY</b><br/><br/>
        This report is generated by an automated quantitative trading system for informational and
        educational purposes only. It should NOT be considered as financial advice or a recommendation
        to buy or sell any securities.<br/><br/>
        <b>Key Points:</b><br/>
        • Past performance does not guarantee future results<br/>
        • All investments carry risk, including potential loss of principal<br/>
        • Market conditions can change rapidly and unpredictably<br/>
        • Machine learning predictions have inherent uncertainties<br/>
        • Technical analysis is not foolproof and should be combined with fundamental analysis<br/><br/>
        <b>Recommendations:</b><br/>
        • Consult with a qualified financial advisor before making investment decisions<br/>
        • Conduct your own research and due diligence<br/>
        • Only invest what you can afford to lose<br/>
        • Diversify your investment portfolio<br/>
        • Regular portfolio rebalancing<br/>
        • Stay informed about market conditions and news<br/><br/>
        <b>Model Limitations:</b><br/>
        • Predictions are based on historical patterns which may not repeat<br/>
        • Unexpected events (black swans) are not accounted for<br/>
        • Market manipulation and insider trading can affect results<br/>
        • Model assumptions may not hold in all market conditions<br/><br/>
        By using this report, you acknowledge that you understand these risks and limitations.<br/><br/>
        <i>Report generated by Advanced Quantitative Trading System with ICT Enhancement</i><br/>
        <i>© 2026 - For Educational Purposes Only</i>
        """
        self.story.append(Paragraph(disclaimer_text, self.styles['Normal']))
# Main Program
def main():
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    print("="*60)
    print("Advanced Quantitative Trading System - ICT Enhanced Edition")
    print("="*60)
    print("This system will demonstrate:")
    print("1️⃣ Data Overview")
    print("2️⃣ Feature Importance Analysis (with ICT)")
    print("3️⃣ Model Training Results")
    print("4️⃣ Complete Technical Analysis (ICT Enhanced)")
    print("5️⃣ Backtest Results Analysis")
    print("6️⃣ FUTURE PRICE PREDICTION")
    print("="*60)
    print("ICT Concepts Included:")
    print(" • Order Blocks (Bullish & Bearish)")
    print(" • Fair Value Gaps (FVG)")
    print(" • Liquidity Sweeps (Stop Hunts)")
    print(" • Market Structure (BOS)")
    print(" • Premium/Discount Zones")
    print(" • Displacement")
    print(" • Breaker Blocks")
    print("="*60 + "\n")

    # Set parameters (customize as needed)
    TICKER = 'AAPL' # Change to '2330.TW' to test TSMC
    START_DATE = '2020-01-01'
    END_DATE = '2025-01-10' # Use latest available data (up to yesterday)
    INITIAL_CAPITAL = 100000
    PREDICT_DAYS = 5 # Number of days to predict into future

    # Data Processing
    processor = AdvancedDataProcessor(TICKER, START_DATE, END_DATE)
    data = processor.fetch_data()
    data = processor.add_advanced_features()

    # Visualization: Data Overview
    plot_data_overview(data, TICKER)

    # Select features (exclude raw prices and target variables)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Target_Return_5d', 'Target_Direction',
                    'Asian_High', 'Asian_Low', 'London_High', 'London_Low'] # Exclude session data
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    print(f"Using {len(feature_cols)} features for training")
    print(f"Feature list: {', '.join([str(col) for col in feature_cols[:10]])}...")

    # Model Training
    predictor = EnsemblePredictor(feature_cols)

    # Use 80% for training, 20% for testing (historical backtest)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    X_train, y_train = predictor.prepare_data(train_data)
    X_test, y_test = predictor.prepare_data(test_data)

    predictor.train_ensemble(X_train, y_train)

    # Visualization: Feature Importance
    plot_feature_importance(predictor, feature_cols)

    # Visualization: Model Training Results
    plot_model_performance(predictor, X_train, y_train, X_test, y_test)

    # Historical Backtest
    test_pred, test_proba = predictor.predict_ensemble(X_test)
    atr_values = test_data['ATR'].values

    # Visualization: Technical Analysis Charts
    plot_technical_analysis(test_data)

    backtester = AdvancedBacktester(initial_capital=INITIAL_CAPITAL)
    metrics = backtester.run_backtest(test_data, test_pred, test_proba, atr_values)

    # Display Backtest Results
    print("\n" + "="*60)
    print("📈 Historical Backtest Results Summary Summary")
    print("="*60)
    print(f"Total Return: {metrics['total_return']:>10.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:>10.2%}")
    print(f"Win Rate: {metrics['win_rate']:>10.2%}")
    print(f"Total Trades: {metrics['total_trades']:>10d}")
    print(f"Avg Win: ${metrics['avg_win']:>10.2f}")
    print(f"Avg Loss: ${metrics['avg_loss']:>10.2f}")
    print("="*60 + "\n")

    # Visualization: Complete Backtest Results Analysis
    plot_backtest_results(metrics, test_data, test_pred, backtester)

    # FUTURE PREDICTION
    print("\n" + "="*60)
    print("NOW PREDICTING FUTURE PRICES...")
    print("="*60)

    # 嘗試載入預訓練權重（假設權重檔案存在於 'saved_models/AAPL_v20260112' 資料夾）
    MODEL_DIR = "/content/drive/MyDrive/saved_models"
    VERSION = datetime.now().strftime("%Y%m%d_%H%M")
    LOAD_PATH = f"{MODEL_DIR}/{TICKER}_{VERSION}"

    print(f"檢查模型儲存路徑：{LOAD_PATH}")

    model_loaded = False

    if os.path.exists(LOAD_PATH) and os.path.exists(f"{LOAD_PATH}/metadata.json"):
        try:
            print("找到完整模型檔案，嘗試載入...")
            predictor.load_models(LOAD_PATH)
            print("模型載入成功！將使用載入的模型進行未來預測。")
            model_loaded = True
        except Exception as e:
            print(f"載入失敗！原因：{str(e)}")
            print("→ 將重新訓練並覆蓋原有模型...")
    else:
        print("未找到完整模型（資料夾或 metadata.json 不存在），將進行完整訓練...")

    # 如果載入失敗 或 本來就沒模型 → 重新訓練
    if not model_loaded:
        print("\n使用完整資料集重新訓練模型...")
        X_all, y_all = predictor.prepare_data(data)
        predictor.train_ensemble(X_all, y_all)
        print("完整訓練完成！")

        # 確保目錄存在
        os.makedirs(LOAD_PATH, exist_ok=True)
        
        print("\n儲存最新訓練完成的模型...")
        predictor.save_models(LOAD_PATH)
        print("模型儲存完成！下次執行可快速載入。")

    # Create future predictor
    future_predictor = FuturePredictor(predictor, processor)

    # Predict future days
    prediction_results, latest_data = future_predictor.predict_future_days(data, n_days=PREDICT_DAYS)

    # Visualize predictions
    future_predictor.plot_future_prediction(data, prediction_results, TICKER)

    # Final Summary
    print("\n" + "="*60)
    print("All analysis complete!")
    print("="*60)
    print("\nSummary:")
    print(f" Historical Performance: {metrics['total_return']:>10.2%} return")
    print(f" Model Confidence: {prediction_results['Confidence'].iloc[-1]:>10.1%}")
    print(f" Future Outlook: {'📈 BULLISH' if prediction_results['Confidence'].iloc[-1] > 0.5 else '📉 BEARISH'}")
    print(f" Expected {PREDICT_DAYS}-Day Change: {((prediction_results['Predicted_Price'].iloc[-1] / prediction_results['Predicted_Price'].iloc[0]) - 1):>9.2%}")

    # Generate PDF Report
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE PDF REPORT")
    print("="*60)

    pdf_filename = f"{TICKER}_Trading_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf_generator = PDFReportGenerator(TICKER, filename=pdf_filename)
    pdf_generator.generate_report(metrics, prediction_results, data, predictor)

    print("\n" + "="*60)
    print("Suggested next steps:")
    print("1. Review the generated PDF report for detailed analysis")
    print("2. Adjust PREDICT_DAYS to forecast further into future")
    print("3. Try different tickers (e.g., 'TSLA', 'MSFT', '2330.TW')")
    print("4. Fine-tune model parameters for better accuracy")
    print("5. Add more ICT concepts or custom indicators")
    print("6. Implement live trading alerts based on predictions")
    print("="*60)

    print(f"\nFiles Generated:")
    print(f" • PDF Report: {pdf_filename}")
    print(f" • Download this file from Colab's Files panel (left sidebar)")

    return metrics, backtester, predictor, prediction_results, pdf_filename
# Execute Main Program
if __name__ == "__main__":
    metrics, backtester, predictor, predictions, pdf_file = main()

    print("\n" + "="*60)
    print("Quick Access to Predictions:")
    print("="*60)
    print("View detailed predictions in the 'predictions' variable:")
    print(predictions.to_string(index=False))
    print("\nProgram execution complete!")
    print(f"\nPDF Report saved as: {pdf_file}")
    print(" You can download it from the Files panel in Colab (icon on the left)")
