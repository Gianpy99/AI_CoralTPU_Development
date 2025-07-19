"""
Technical signal generator for trading decisions
"""

from typing import List, Dict, Any
from loguru import logger
import numpy as np

from src.models.data.market_data import MarketData, TradingSignal


class TechnicalSignalGenerator:
    """Generates trading signals based on technical analysis"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 ma_short: int = 9,
                 ma_long: int = 21):
        self.rsi_period = rsi_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        
    def generate_signal(self, market_data: List[MarketData]) -> TradingSignal:
        """Generate trading signal from market data"""
        
        if not market_data:
            return TradingSignal(
                timestamp=int(time.time()),
                symbol="UNKNOWN",
                action="hold",
                confidence=0.0,
                reason="No market data available"
            )
        
        latest = market_data[-1]
        
        # Simple demo signal based on price change
        if len(market_data) >= 2:
            prev = market_data[-2]
            price_change = (latest.close - prev.close) / prev.close
            
            if price_change > 0.02:  # 2% increase
                return TradingSignal(
                    timestamp=latest.timestamp,
                    symbol=latest.symbol,
                    action="buy",
                    confidence=min(abs(price_change) * 10, 0.8),
                    price=latest.close,
                    reason=f"Price increased by {price_change:.2%}"
                )
            elif price_change < -0.02:  # 2% decrease
                return TradingSignal(
                    timestamp=latest.timestamp,
                    symbol=latest.symbol,
                    action="sell",
                    confidence=min(abs(price_change) * 10, 0.8),
                    price=latest.close,
                    reason=f"Price decreased by {price_change:.2%}"
                )
        
        # Default to hold
        return TradingSignal(
            timestamp=latest.timestamp,
            symbol=latest.symbol,
            action="hold",
            confidence=0.5,
            price=latest.close,
            reason="No clear signal"
        )
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_moving_average(self, prices: List[float], period: int) -> float:
        """Calculate simple moving average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        return np.mean(prices[-period:])


# Import time for timestamp
import time
