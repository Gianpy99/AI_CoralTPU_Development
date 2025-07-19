"""
Configuration management for the Coral TPU Trading System
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv


class Settings:
    """Application settings loaded from environment variables"""
    
    def __init__(self):
        # Load environment variables from .env file
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Exchange API Configuration
        self.BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
        self.BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
        self.COINBASE_API_KEY: str = os.getenv("COINBASE_API_KEY", "")
        self.COINBASE_SECRET_KEY: str = os.getenv("COINBASE_SECRET_KEY", "")
        
        # Data Sources
        self.ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        
        # Trading Configuration
        self.TRADING_MODE: str = os.getenv("TRADING_MODE", "simulation")
        self.trading_pair: str = os.getenv("TRADING_PAIR", "BTCUSDT")
        self.DEFAULT_SYMBOLS: List[str] = os.getenv("DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
        self.MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.02"))
        self.STOP_LOSS_PERCENTAGE: float = float(os.getenv("STOP_LOSS_PERCENTAGE", "0.02"))
        self.TAKE_PROFIT_PERCENTAGE: float = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "0.05"))
        
        # Model Configuration
        self.MODEL_PATH: str = os.getenv("MODEL_PATH", "models/price_predictor.tflite")
        self.CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        self.PREDICTION_HORIZON: int = int(os.getenv("PREDICTION_HORIZON", "5"))
        
        # System Configuration
        self.LOOP_INTERVAL: int = int(os.getenv("LOOP_INTERVAL", "30"))
        self.ENABLE_DASHBOARD: bool = os.getenv("ENABLE_DASHBOARD", "true").lower() == "true"
        self.DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8000"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Notifications
        self.TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
        self.ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
        
        # Database
        self.DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/trading_data.db")
        
        # Risk Management
        self.MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
        self.MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", "0.10"))
        self.POSITION_SIZING_METHOD: str = os.getenv("POSITION_SIZING_METHOD", "kelly")
        
        # Paths
        self.DATA_DIR: Path = Path("data")
        self.MODELS_DIR: Path = Path("models")
        self.LOGS_DIR: Path = Path("logs")
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check required API keys for live trading
        if self.TRADING_MODE == "live":
            if not self.BINANCE_API_KEY or not self.BINANCE_SECRET_KEY:
                errors.append("Binance API keys are required for live trading")
        
        # Check model file exists
        if not Path(self.MODEL_PATH).exists() and self.TRADING_MODE != "simulation":
            errors.append(f"Model file not found: {self.MODEL_PATH}")
        
        # Check value ranges
        if not 0 < self.MAX_POSITION_SIZE <= 1:
            errors.append("MAX_POSITION_SIZE must be between 0 and 1")
        
        if not 0 < self.CONFIDENCE_THRESHOLD <= 1:
            errors.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
