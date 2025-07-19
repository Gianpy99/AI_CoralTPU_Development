"""
Market data models for the trading system
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketData:
    """Represents market data for a trading symbol"""
    
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.high < self.low:
            raise ValueError("High price cannot be less than low price")
        if self.close < 0 or self.open < 0:
            raise ValueError("Prices cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def price_change(self) -> float:
        """Calculate price change from open to close"""
        return self.close - self.open
    
    @property
    def price_change_percent(self) -> float:
        """Calculate percentage price change"""
        if self.open == 0:
            return 0.0
        return (self.price_change / self.open) * 100
    
    @property
    def range_percent(self) -> float:
        """Calculate the price range as percentage of open"""
        if self.open == 0:
            return 0.0
        return ((self.high - self.low) / self.open) * 100


@dataclass
class TradingSignal:
    """Represents a trading signal"""
    
    timestamp: int
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: Optional[str] = None
    
    def __post_init__(self):
        """Validate signal data"""
        if self.action not in ['buy', 'sell', 'hold']:
            raise ValueError("Action must be 'buy', 'sell', or 'hold'")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class Position:
    """Represents a trading position"""
    
    symbol: str
    side: str  # 'long', 'short'
    size: float
    entry_price: float
    current_price: float
    timestamp: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss"""
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.size
        else:  # short
            return (self.entry_price - self.current_price) * self.size
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized profit/loss percentage"""
        if self.entry_price == 0:
            return 0.0
        if self.side == 'long':
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
