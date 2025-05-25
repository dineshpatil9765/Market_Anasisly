import streamlit as st
from streamlit_autorefresh import st_autorefresh
from binance.client import Client
import pandas as pd
import ta
import threading
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from pathlib import Path
import time
import os
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = Path('data')
MODEL_DIR = Path('models')
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Configuration
class Config:
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'your_api_key_here')
    BINANCE_SECRET = os.getenv('BINANCE_SECRET', 'your_secret_here')
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', 'your_newsapi_key')
    REFRESH_INTERVAL = 30000  # 30 seconds in milliseconds
    DEFAULT_SYMBOL = 'BTCUSDT'
    DEFAULT_INTERVAL = '5m'
    DEFAULT_BALANCE = 1000.0
    DEFAULT_RISK = 1.0  # 1% risk per trade
    MAX_HISTORY_ENTRIES = 1000  # Maximum number of signals to keep in history

# Alert System
class AlertSystem:
    @staticmethod
    def play_alert():
        try:
            from playsound import playsound
            playsound("alert.mp3")
        except:
            st.warning("Could not play alert sound. Make sure 'alert.mp3' is in the directory.")

# Data Storage
class DataStorage:
    @staticmethod
    def get_data_filename(symbol, interval):
        return DATA_DIR / f"{symbol}_{interval}_data.csv"
    
    @staticmethod
    def save_to_csv(df, symbol, interval):
        filename = DataStorage.get_data_filename(symbol, interval)
        df.to_csv(filename, index=False)
    
    @staticmethod
    def load_historical_data(symbol, interval):
        filename = DataStorage.get_data_filename(symbol, interval)
        try:
            df = pd.read_csv(filename, parse_dates=['time'])
            # Ensure numeric columns are numeric
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            return df.dropna()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

# Signal History Management
class SignalHistory:
    @staticmethod
    def get_history_filename():
        return DATA_DIR / "signal_history.csv"
    
    @staticmethod
    def load_history():
        filename = SignalHistory.get_history_filename()
        try:
            history = pd.read_csv(filename, parse_dates=['timestamp'])
            # Ensure numeric columns are numeric
            numeric_cols = ['confidence', 'price', 'stop_loss', 'take_profit']
            history[numeric_cols] = history[numeric_cols].apply(pd.to_numeric, errors='coerce')
            return history.dropna()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=['timestamp', 'symbol', 'interval', 'signal', 
                                       'confidence', 'price', 'stop_loss', 'take_profit'])
    
    @staticmethod
    def save_signal(signal_data):
        filename = SignalHistory.get_history_filename()
        history = SignalHistory.load_history()
        
        # Limit history size
        if len(history) >= Config.MAX_HISTORY_ENTRIES:
            history = history.iloc[-(Config.MAX_HISTORY_ENTRIES-1):]
        
        new_entry = pd.DataFrame([signal_data])
        updated_history = pd.concat([history, new_entry], ignore_index=True)
        
        updated_history.to_csv(filename, index=False)

# Candlestick Pattern Detection
class PatternDetector:
    @staticmethod
    def is_doji(candle):
        body = abs(candle['Close'] - candle['Open'])
        range_ = candle['High'] - candle['Low']
        return body < 0.1 * range_ if range_ != 0 else False

    @staticmethod
    def is_hammer(candle):
        body = abs(candle['Close'] - candle['Open'])
        lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
        upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
        return (lower_wick > 2 * body) and (upper_wick < body) if body != 0 else False

    @staticmethod
    def is_bullish_engulfing(prev, curr):
        return (curr['Open'] < curr['Close']) and \
               (prev['Open'] > prev['Close']) and \
               (curr['Open'] < prev['Close']) and \
               (curr['Close'] > prev['Open'])

    @staticmethod
    def is_bearish_engulfing(prev, curr):
        return (curr['Open'] > curr['Close']) and \
               (prev['Open'] < prev['Close']) and \
               (curr['Open'] > prev['Close']) and \
               (curr['Close'] < prev['Open'])

    @staticmethod
    def detect_patterns(df, window=20):
        patterns = []
        for i in range(1, min(window, len(df))):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            pattern = None
            signal = None
            
            if PatternDetector.is_doji(candle): 
                pattern = 'Doji'
                signal = 'NEUTRAL'
            elif PatternDetector.is_hammer(candle): 
                pattern = 'Hammer'
                signal = 'BUY'
            elif PatternDetector.is_bullish_engulfing(prev_candle, candle): 
                pattern = 'Bullish Engulfing'
                signal = 'BUY'
            elif PatternDetector.is_bearish_engulfing(prev_candle, candle):
                pattern = 'Bearish Engulfing'
                signal = 'SELL'
            
            if pattern:
                patterns.append({
                    'time': candle['time'],
                    'pattern': pattern,
                    'close': candle['Close'],
                    'signal': signal
                })
        return patterns

# Technical Analysis
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df):
        # Ensure we have enough data
        if len(df) < 50:
            return df
        
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Oscillators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # Volatility
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        # Volume
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
            df['High'], df['Low'], df['Close'], df['Volume'], window=20).volume_weighted_average_price()
        
        return df.dropna()

# Machine Learning Model
class TradingModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = MODEL_DIR / 'trading_model.pkl'
        self.scaler_path = MODEL_DIR / 'scaler.pkl'
        
    def train_model(self, df):
        if len(df) < 100:
            st.warning("Not enough data to train model. Need at least 100 samples.")
            return False
        
        features = ['SMA_10', 'SMA_20', 'RSI', 'MACD', 'ADX', 'ATR', 'VWAP']
        X = df[features].values[:-1]
        y = (df['Close'].values[1:] > df['Close'].values[:-1]).astype(int)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_scaled, y)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return True
    
    def load_model(self):
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                return True
            except:
                return False
        return False
    
    def predict(self, df):
        if not self.model or not self.scaler:
            if not self.load_model():
                return "HOLD", 50
        
        features = ['SMA_10', 'SMA_20', 'RSI', 'MACD', 'ADX', 'ATR', 'VWAP']
        if not all(f in df.columns for f in features):
            return "HOLD", 50
            
        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        latest_scaled = self.scaler.transform(latest_features)
        
        proba = self.model.predict_proba(latest_scaled)[0]
        confidence = int(max(proba) * 100)
        
        if proba[1] > 0.65: return 'BUY', confidence
        elif proba[0] > 0.65: return 'SELL', confidence
        return 'HOLD', confidence

# Risk Management
class RiskManager:
    @staticmethod
    def calculate_position_size(account_balance, risk_per_trade_pct, entry_price, stop_loss_price, atr):
        volatility_factor = 1 / (1 + (atr / entry_price)) if entry_price != 0 else 1
        risk_amount = account_balance * (risk_per_trade_pct / 100) * volatility_factor
        stop_loss_risk = abs(entry_price - stop_loss_price)
        return risk_amount / stop_loss_risk if stop_loss_risk != 0 else 0
    
    @staticmethod
    def calculate_dynamic_stop_loss(df):
        if len(df) < 5:
            return df['Low'].iloc[-1], df['High'].iloc[-1]
            
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else df['High'].iloc[-1] - df['Low'].iloc[-1]
        latest_close = df['Close'].iloc[-1]
        stop_distance = min(max(atr, latest_close * 0.005), latest_close * 0.05)
        recent_low = df['Low'].iloc[-5:].min()
        recent_high = df['High'].iloc[-5:].max()
        return recent_low - (stop_distance * 0.5), recent_high + (stop_distance * 0.5)

# Data Fetching
class DataFetcher:
    @staticmethod
    def fetch_binance_data(symbol=Config.DEFAULT_SYMBOL, interval=Config.DEFAULT_INTERVAL, limit=200):
        try:
            client = Client(Config.BINANCE_API_KEY, Config.BINANCE_SECRET)
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            df = pd.DataFrame(klines, columns=[
                "Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "Quote Asset Volume", "Number of Trades",
                "Taker Buy Base", "Taker Buy Quote", "Ignore"])
            
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].astype(float)
            
            df['time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df = df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Calculate technical indicators
            df = TechnicalAnalyzer.calculate_indicators(df)
            
            # Save data
            DataStorage.save_to_csv(df, symbol, interval)
            
            return df
        except Exception as e:
            st.error(f"Error fetching data from Binance: {str(e)}")
            return pd.DataFrame()

# Signal Generation
class SignalGenerator:
    @staticmethod
    def generate_signal(df, ml_model):
        if len(df) < 20:
            return "HOLD", 50
            
        df = TechnicalAnalyzer.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Technical Conditions
        bullish_conditions = [
            latest['SMA_10'] > latest['SMA_20'],
            latest['EMA_12'] > latest['EMA_26'],
            50 < latest['RSI'] < 70,
            latest['MACD'] > 0,
            latest['Close'] > latest['VWAP'],
            latest['ADX'] > 25,
            latest['Close'] > latest['BB_middle']
        ]
        
        bearish_conditions = [
            latest['SMA_10'] < latest['SMA_20'],
            latest['EMA_12'] < latest['EMA_26'],
            30 < latest['RSI'] < 50,
            latest['MACD'] < 0,
            latest['Close'] < latest['VWAP'],
            latest['ADX'] > 25,
            latest['Close'] < latest['BB_middle']
        ]
        
        # Pattern detection
        patterns = PatternDetector.detect_patterns(df.iloc[-10:])
        recent_bullish_patterns = sum(1 for p in patterns if p['signal'] == 'BUY')
        recent_bearish_patterns = sum(1 for p in patterns if p['signal'] == 'SELL')
        
        # ML prediction
        ml_signal, ml_confidence = ml_model.predict(df)
        
        # Scoring
        technical_score = sum(bullish_conditions) - sum(bearish_conditions)
        pattern_score = recent_bullish_patterns - recent_bearish_patterns
        
        # Decision making
        if technical_score >= 4 and pattern_score > 0 and ml_signal == 'BUY':
            return 'BUY', min(90, ml_confidence + 10)
        elif technical_score <= -4 and pattern_score < 0 and ml_signal == 'SELL':
            return 'SELL', min(90, ml_confidence + 10)
        elif ml_confidence > 75 and ml_signal in ['BUY', 'SELL']:
            return ml_signal, ml_confidence
        return 'HOLD', max(50, ml_confidence)

# News Integration
class NewsAnalyzer:
    @staticmethod
    def get_crypto_news(symbol='bitcoin', limit=5):
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={Config.NEWSAPI_KEY}&pageSize={limit}"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                return [{
                    'title': a['title'],
                    'source': a['source']['name'],
                    'url': a['url'],
                    'published_at': pd.to_datetime(a['publishedAt']).strftime('%Y-%m-%d %H:%M')
                } for a in articles]
        except:
            return []
        return []

# Visualization
class ChartRenderer:
    @staticmethod
    def render_candlestick_chart(df, patterns=None):
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Moving Averages
        for ma in ['SMA_10', 'SMA_20', 'SMA_50']:
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df[ma],
                    name=ma,
                    line=dict(width=1)
                ))
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['BB_upper'],
                name='BB Upper',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['BB_lower'],
                name='BB Lower',
                fill='tonexty',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1)
            ))
        
        # Patterns
        if patterns:
            for pattern in patterns:
                color = 'green' if pattern['signal'] == 'BUY' else 'red' if pattern['signal'] == 'SELL' else 'gray'
                fig.add_annotation(
                    x=pattern['time'],
                    y=pattern['close'],
                    text=pattern['pattern'],
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    bgcolor=color,
                    opacity=0.8
                )
        
        fig.update_layout(
            title='Price Chart with Indicators',
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def render_technical_indicators(df):
        if len(df) < 20:
            return None
            
        fig = go.Figure()
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        fig.add_hline(y=70, line_dash="dot", line_color="red")
        fig.add_hline(y=30, line_dash="dot", line_color="green")
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['MACD'],
                name='MACD',
                line=dict(color='orange', width=2)
            ))
            fig.add_hline(y=0, line_dash="solid", line_color="white")
        
        # Volume
        fig.add_trace(go.Bar(
            x=df['time'],
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(100, 100, 200, 0.5)'
        ))
        
        fig.update_layout(
            title='Technical Indicators',
            xaxis_title='Time',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def render_signal_history(history):
        if len(history) < 1:
            return None
            
        fig = go.Figure()
        
        # Add buy signals
        buy_signals = history[history['signal'] == 'BUY']
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['price'],
                mode='markers',
                name='Buy Signals',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        # Add sell signals
        sell_signals = history[history['signal'] == 'SELL']
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['price'],
                mode='markers',
                name='Sell Signals',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
        
        fig.update_layout(
            title='Signal History',
            xaxis_title='Time',
            yaxis_title='Price',
            height=400,
            template='plotly_dark'
        )
        
        return fig

# Main App
class CryptoDashboard:
    def __init__(self):
        # Initialize page config first
        st.set_page_config(
            page_title="Advanced Crypto Trading Dashboard",
            layout="wide",
            page_icon="📊"
        )
        
        self.ml_model = TradingModel()
        self.data = pd.DataFrame()
        self.patterns = []
        self.signal = "HOLD"
        self.confidence = 50
        self.position_size = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Initialize settings with default values
        self.symbol = Config.DEFAULT_SYMBOL
        self.interval = '5m'
        self.account_balance = Config.DEFAULT_BALANCE
        self.risk_pct = Config.DEFAULT_RISK
        
    def run(self):
        # Auto-refresh
        st_autorefresh(interval=Config.REFRESH_INTERVAL, key="refresh")
        
        # First get settings from sidebar
        self.setup_sidebar()
        
        # Then fetch data with current settings
        self.fetch_data()
        
        # Main content
        st.title("📊 Advanced Crypto Trading Dashboard")
        
        if not self.data.empty:
            self.display_metrics()
            self.display_charts()
            self.display_signal_history()
            self.display_news()
            self.display_trade_recommendation()
        else:
            st.warning("No data available. Check your API connection and symbol.")
    
    def setup_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Dashboard Settings")
            
            # Symbol input
            self.symbol = st.text_input(
                "Cryptocurrency Pair", 
                value=Config.DEFAULT_SYMBOL,
                help="Format: BTCUSDT, ETHUSDT, etc."
            ).upper()
            
            # Time interval
            self.interval = st.selectbox(
                "Time Interval",
                options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=2
            )
            
            # Account settings
            st.subheader("💰 Account Settings")
            self.account_balance = st.number_input(
                "Account Balance (USD)",
                value=Config.DEFAULT_BALANCE,
                min_value=10.0,
                step=100.0
            )
            
            self.risk_pct = st.slider(
                "Risk per Trade (%)",
                min_value=0.1,
                max_value=10.0,
                value=Config.DEFAULT_RISK,
                step=0.1
            )
            
            # Model controls
            st.subheader("🤖 Model Controls")
            if st.button("Retrain Machine Learning Model"):
                with st.spinner("Training model..."):
                    if self.ml_model.train_model(self.data):
                        st.success("Model trained successfully!")
                    else:
                        st.error("Failed to train model. Not enough data?")
            
            # Data management
            st.subheader("🔄 Data Management")
            if st.button("Force Data Refresh"):
                self.fetch_data()
            
            # Signal history controls
            st.subheader("📈 Signal History")
            if st.button("Clear Signal History"):
                SignalHistory.save_signal(pd.DataFrame(columns=['timestamp', 'symbol', 'interval', 'signal', 
                                       'confidence', 'price', 'stop_loss', 'take_profit']))
                st.success("Signal history cleared!")
    
    def fetch_data(self):
        with st.spinner("Fetching latest data..."):
            # Load historical data
            historical_data = DataStorage.load_historical_data(self.symbol, self.interval)
            
            # Fetch new data
            new_data = DataFetcher.fetch_binance_data(self.symbol, self.interval)
            
            if not new_data.empty:
                # Combine with historical data
                if not historical_data.empty:
                    self.data = pd.concat([historical_data, new_data]).drop_duplicates('time', keep='last')
                else:
                    self.data = new_data
                
                # Calculate indicators
                self.data = TechnicalAnalyzer.calculate_indicators(self.data)
                
                # Detect patterns
                self.patterns = PatternDetector.detect_patterns(self.data)
                
                # Generate signal
                self.signal, self.confidence = SignalGenerator.generate_signal(self.data, self.ml_model)
                
                # Save signal to history if not HOLD
                if self.signal != 'HOLD':
                    signal_data = {
                        'timestamp': datetime.now(),
                        'symbol': self.symbol,
                        'interval': self.interval,
                        'signal': self.signal,
                        'confidence': self.confidence,
                        'price': self.data['Close'].iloc[-1],
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    }
                    SignalHistory.save_signal(signal_data)
                
                # Calculate position size
                buy_stop, sell_stop = RiskManager.calculate_dynamic_stop_loss(self.data)
                entry_price = self.data['Close'].iloc[-1]
                atr = self.data['ATR'].iloc[-1] if 'ATR' in self.data.columns else entry_price * 0.01
                
                if self.signal == 'BUY':
                    self.position_size = RiskManager.calculate_position_size(
                        self.account_balance, self.risk_pct, entry_price, buy_stop, atr)
                    self.stop_loss = buy_stop
                    self.take_profit = entry_price + 2 * (entry_price - buy_stop)
                elif self.signal == 'SELL':
                    self.position_size = RiskManager.calculate_position_size(
                        self.account_balance, self.risk_pct, entry_price, sell_stop, atr)
                    self.stop_loss = sell_stop
                    self.take_profit = entry_price - 2 * (sell_stop - entry_price)
                else:
                    self.position_size = 0
                    self.stop_loss = 0
                    self.take_profit = 0
            else:
                st.error("Failed to fetch new data. Using cached data if available.")
                self.data = historical_data
    
    def display_metrics(self):
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2] if len(self.data) > 1 else latest
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Price metric
        price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        col1.metric(
            "Current Price", 
            f"${latest['Close']:.2f}", 
            delta=f"{price_change:.2f}%"
        )
        
        # RSI metric
        rsi_status = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"
        rsi_color = "red" if latest['RSI'] > 70 else "green" if latest['RSI'] < 30 else "gray"
        col2.metric(
            "RSI (14)", 
            f"{latest['RSI']:.2f}", 
            rsi_status,
            delta_color="off"
        )
        
        # Volatility metric
        col3.metric(
            "Volatility (ATR)", 
            f"${latest['ATR']:.2f}" if 'ATR' in latest else "N/A"
        )
        
        # Signal display
        signal_color = "green" if self.signal == "BUY" else "red" if self.signal == "SELL" else "gray"
        col4.markdown(f"""
        <div style="border:2px solid {signal_color}; padding:10px; border-radius:5px;">
            <h2 style="color:{signal_color}; text-align:center;">{self.signal}</h2>
            <div style="text-align:center; padding:5px;">
                Confidence: {self.confidence}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Play alert if strong signal
        if self.signal != "HOLD" and self.confidence > 70:
            threading.Thread(target=AlertSystem.play_alert).start()
    
    def display_charts(self):
        # Main price chart
        st.plotly_chart(
            ChartRenderer.render_candlestick_chart(self.data, self.patterns),
            use_container_width=True
        )
        
        # Technical indicators
        tech_chart = ChartRenderer.render_technical_indicators(self.data)
        if tech_chart:
            st.plotly_chart(tech_chart, use_container_width=True)
    
    def display_signal_history(self):
        with st.expander("📈 Signal History", expanded=True):
            history = SignalHistory.load_history()
            
            if not history.empty:
                # Filter for current symbol
                symbol_history = history[history['symbol'] == self.symbol]
                
                if not symbol_history.empty:
                    # Display chart
                    history_chart = ChartRenderer.render_signal_history(symbol_history)
                    if history_chart:
                        st.plotly_chart(history_chart, use_container_width=True)
                    
                    # Convert to display format
                    display_df = symbol_history.sort_values('timestamp', ascending=False)
                    display_df = display_df[['timestamp', 'signal', 'confidence', 
                                           'price', 'stop_loss', 'take_profit']]
                    display_df.columns = ['Time', 'Signal', 'Confidence', 
                                        'Price', 'Stop Loss', 'Take Profit']
                    
                    # Format numbers
                    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
                    display_df['Stop Loss'] = display_df['Stop Loss'].apply(lambda x: f"${x:.2f}")
                    display_df['Take Profit'] = display_df['Take Profit'].apply(lambda x: f"${x:.2f}")
                    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x}%")
                    
                    st.dataframe(display_df, height=300)
                    
                    # Download button
                    csv = symbol_history.to_csv(index=False)
                    st.download_button(
                        label="Download Full History",
                        data=csv,
                        file_name=f"{self.symbol}_signal_history.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No signal history for this symbol yet.")
            else:
                st.info("No signal history recorded yet.")
    
    def display_news(self):
        news = NewsAnalyzer.get_crypto_news(self.symbol[:-4].lower(), limit=3)
        if news:
            with st.expander("📰 Latest Crypto News"):
                for item in news:
                    st.markdown(f"""
                    **{item['title']}**  
                    *{item['source']} - {item['published_at']}*  
                    [Read more]({item['url']})
                    """)
                    st.write("---")
    
    def display_trade_recommendation(self):
        if self.signal != "HOLD":
            with st.expander(f"🚨 Trade Recommendation ({self.signal})", expanded=True):
                current_price = self.data['Close'].iloc[-1]
                
                st.subheader(f"Recommended Action: {self.signal}")
                st.write(f"**Confidence Level:** {self.confidence}%")
                st.write(f"**Current Price:** ${current_price:.2f}")
                
                if self.signal == "BUY":
                    st.write(f"**Recommended Stop Loss:** ${self.stop_loss:.2f}")
                    st.write(f"**Recommended Take Profit:** ${self.take_profit:.2f}")
                else:  # SELL
                    st.write(f"**Recommended Stop Loss:** ${self.stop_loss:.2f}")
                    st.write(f"**Recommended Take Profit:** ${self.take_profit:.2f}")
                
                st.write(f"**Position Size:** {self.position_size:.4f} {self.symbol[:-4]}")
                st.write(f"**Risk Amount:** ${self.account_balance * (self.risk_pct / 100):.2f}")
                
                st.progress(self.confidence / 100)
                
                if st.button("📢 Send Alert Again"):
                    AlertSystem.play_alert()

# Run the app
if __name__ == "__main__":
    app = CryptoDashboard()
    app.fetch_data()  # Initial data load
    app.run()