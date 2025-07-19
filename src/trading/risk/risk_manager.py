"""
Risk management system for trading operations
"""

from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class RiskLimits:
    """Risk management limits"""
    max_position_size: float = 0.1  # 10% of account
    max_daily_loss: float = 0.05    # 5% daily loss limit
    max_drawdown: float = 0.15      # 15% maximum drawdown
    stop_loss_percentage: float = 0.02  # 2% stop loss
    take_profit_percentage: float = 0.05  # 5% take profit


class RiskManager:
    """Manages trading risk and position sizing"""
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_daily_loss: float = 0.05,
                 stop_loss_percentage: float = 0.02,
                 take_profit_percentage: float = 0.05):
        
        self.limits = RiskLimits(
            max_position_size=max_position_size,
            max_daily_loss=max_daily_loss,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage
        )
        
        self.daily_pnl = 0.0
        self.daily_reset_time = None
        
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float,
                              stop_loss_price: Optional[float] = None) -> float:
        """Calculate position size based on risk management rules"""
        
        # Maximum position size based on account percentage
        max_position_value = account_balance * self.limits.max_position_size
        max_quantity_by_account = max_position_value / entry_price
        
        # If we have a stop loss, calculate position size based on risk
        if stop_loss_price and stop_loss_price != entry_price:
            risk_per_unit = abs(entry_price - stop_loss_price)
            max_risk_amount = account_balance * self.limits.stop_loss_percentage
            max_quantity_by_risk = max_risk_amount / risk_per_unit
            
            # Use the smaller of the two
            quantity = min(max_quantity_by_account, max_quantity_by_risk)
        else:
            quantity = max_quantity_by_account
        
        logger.info(f"Calculated position size: {quantity:.6f} (max by account: {max_quantity_by_account:.6f})")
        return quantity
    
    def validate_trade(self, 
                      symbol: str,
                      side: str,
                      quantity: float,
                      price: float,
                      account_balance: float) -> bool:
        """Validate if a trade meets risk management criteria"""
        
        position_value = quantity * price
        position_percentage = position_value / account_balance
        
        # Check position size limit
        if position_percentage > self.limits.max_position_size:
            logger.warning(f"Position size {position_percentage:.2%} exceeds limit {self.limits.max_position_size:.2%}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -account_balance * self.limits.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False
        
        logger.info(f"Trade validation passed for {side} {quantity} {symbol}")
        return True
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side.lower() == "buy" or side.lower() == "long":
            return entry_price * (1 - self.limits.stop_loss_percentage)
        else:  # sell or short
            return entry_price * (1 + self.limits.stop_loss_percentage)
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side.lower() == "buy" or side.lower() == "long":
            return entry_price * (1 + self.limits.take_profit_percentage)
        else:  # sell or short
            return entry_price * (1 - self.limits.take_profit_percentage)
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl_change
        logger.debug(f"Daily P&L updated: {self.daily_pnl:.2f}")
    
    def reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.daily_pnl = 0.0
        logger.info("Daily risk metrics reset")
    
    def get_risk_summary(self, account_balance: float) -> Dict:
        """Get current risk status summary"""
        return {
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percentage": (self.daily_pnl / account_balance) * 100,
            "daily_loss_limit": account_balance * self.limits.max_daily_loss,
            "max_position_size": account_balance * self.limits.max_position_size,
            "stop_loss_percentage": self.limits.stop_loss_percentage * 100,
            "take_profit_percentage": self.limits.take_profit_percentage * 100,
            "risk_limits": {
                "max_position_size": self.limits.max_position_size * 100,
                "max_daily_loss": self.limits.max_daily_loss * 100,
                "max_drawdown": self.limits.max_drawdown * 100
            }
        }
    
    def check_emergency_stop(self, current_drawdown: float) -> bool:
        """Check if emergency stop should be triggered"""
        if current_drawdown >= self.limits.max_drawdown:
            logger.critical(f"Emergency stop triggered! Drawdown {current_drawdown:.2%} >= limit {self.limits.max_drawdown:.2%}")
            return True
        return False
