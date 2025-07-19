"""
Portfolio management for tracking balances and positions
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from src.models.data.market_data import Position


@dataclass
class Balance:
    """Represents account balance"""
    currency: str
    free: float
    locked: float = 0.0
    
    @property
    def total(self) -> float:
        return self.free + self.locked


class PortfolioManager:
    """Manages trading portfolio including balances and positions"""
    
    def __init__(self, initial_balance: float = 10000.0, base_currency: str = "USDT"):
        self.base_currency = base_currency
        self.balances: Dict[str, Balance] = {
            base_currency: Balance(currency=base_currency, free=initial_balance)
        }
        self.positions: Dict[str, Position] = {}
        self.initial_balance = initial_balance
        
    def get_balance(self, currency: str) -> Balance:
        """Get balance for a specific currency"""
        return self.balances.get(currency, Balance(currency=currency, free=0.0))
    
    def get_total_balance(self) -> float:
        """Get total portfolio value in base currency"""
        total = 0.0
        for balance in self.balances.values():
            if balance.currency == self.base_currency:
                total += balance.total
            else:
                # In a real system, we'd convert to base currency using current prices
                # For now, we'll just include base currency
                pass
        return total
    
    def update_balance(self, currency: str, free: float, locked: float = 0.0):
        """Update balance for a currency"""
        self.balances[currency] = Balance(
            currency=currency,
            free=free,
            locked=locked
        )
        
    def add_position(self, position: Position):
        """Add a new position"""
        self.positions[position.symbol] = position
        logger.info(f"Added position: {position.side} {position.size} {position.symbol} @ {position.entry_price}")
        
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position"""
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            logger.info(f"Removed position: {symbol}")
            return position
        return None
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for a position"""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
    
    def get_total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L across all positions"""
        return sum(position.unrealized_pnl for position in self.positions.values())
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        total_balance = self.get_total_balance()
        total_pnl = self.get_total_unrealized_pnl()
        
        return {
            "total_balance": total_balance,
            "initial_balance": self.initial_balance,
            "realized_pnl": total_balance - self.initial_balance,
            "unrealized_pnl": total_pnl,
            "total_pnl": (total_balance - self.initial_balance) + total_pnl,
            "pnl_percentage": ((total_balance - self.initial_balance) / self.initial_balance) * 100,
            "number_of_positions": len(self.positions),
            "balances": {currency: balance.total for currency, balance in self.balances.items()}
        }
    
    def can_afford(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if we can afford a trade"""
        required_amount = quantity * price
        base_balance = self.get_balance(self.base_currency)
        
        return base_balance.free >= required_amount
    
    def execute_trade(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Execute a trade and update balances"""
        
        cost = quantity * price
        
        if side == "buy":
            base_balance = self.get_balance(self.base_currency)
            if base_balance.free < cost:
                logger.warning(f"Insufficient balance for {side} {quantity} {symbol}")
                return False
            
            # Update balances
            self.update_balance(
                self.base_currency,
                base_balance.free - cost,
                base_balance.locked
            )
            
            # Add or update asset balance
            asset_currency = symbol.replace(self.base_currency, "")
            asset_balance = self.get_balance(asset_currency)
            self.update_balance(
                asset_currency,
                asset_balance.free + quantity,
                asset_balance.locked
            )
            
        elif side == "sell":
            asset_currency = symbol.replace(self.base_currency, "")
            asset_balance = self.get_balance(asset_currency)
            
            if asset_balance.free < quantity:
                logger.warning(f"Insufficient {asset_currency} for {side} {quantity} {symbol}")
                return False
            
            # Update balances
            self.update_balance(
                asset_currency,
                asset_balance.free - quantity,
                asset_balance.locked
            )
            
            base_balance = self.get_balance(self.base_currency)
            self.update_balance(
                self.base_currency,
                base_balance.free + cost,
                base_balance.locked
            )
        
        logger.info(f"Executed {side} {quantity} {symbol} @ {price}")
        return True
