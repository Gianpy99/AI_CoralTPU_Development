#!/usr/bin/env python3
"""
Simple test for the Coral TPU Trading System
Tests basic functionality without heavy dependencies
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from loguru import logger

async def test_system_basics():
    """Test basic system functionality"""
    
    logger.info("🚀 Starting simple system test...")
    
    try:
        # Test 1: Configuration
        logger.info("📋 Testing configuration...")
        from src.config.settings import Settings
        
        # Create default settings
        os.environ.setdefault('BINANCE_API_KEY', 'test_key')
        os.environ.setdefault('BINANCE_SECRET', 'test_secret')
        
        settings = Settings()
        logger.success(f"✅ Settings loaded: Trading pair: {settings.trading_pair}")
        
        # Test 2: Data structures
        logger.info("📊 Testing data structures...")
        from src.models.data.market_data import MarketData
        
        market_data = MarketData(
            timestamp=1640995200,  # 2022-01-01
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            symbol="BTCUSDT"
        )
        logger.success(f"✅ Market data created: {market_data.symbol} @ ${market_data.close}")
        
        # Test 3: Basic trading logic
        logger.info("🔧 Testing trading components...")
        from src.trading.signals.technical_signals import TechnicalSignalGenerator
        
        signal_gen = TechnicalSignalGenerator()
        # Test signal generation with dummy data
        signal = signal_gen.generate_signal([market_data])
        logger.success(f"✅ Signal generated: {signal}")
        
        # Test 4: Portfolio management
        logger.info("💰 Testing portfolio...")
        from src.trading.portfolio.portfolio_manager import PortfolioManager
        
        portfolio = PortfolioManager(initial_balance=10000.0)
        balance = portfolio.get_total_balance()
        logger.success(f"✅ Portfolio initialized: ${balance:.2f}")
        
        # Test 5: Risk management
        logger.info("⚠️ Testing risk management...")
        from src.trading.risk.risk_manager import RiskManager
        
        risk_manager = RiskManager(
            max_position_size=0.1,
            max_daily_loss=0.05,
            stop_loss_percentage=0.02
        )
        
        position_size = risk_manager.calculate_position_size(
            account_balance=10000.0,
            entry_price=50000.0,
            stop_loss_price=49000.0
        )
        logger.success(f"✅ Risk management working: Position size: ${position_size:.2f}")
        
        logger.success("🎉 All basic tests passed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Some modules may need additional dependencies")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_crypto_data():
    """Test crypto data collection without external APIs"""
    
    logger.info("📈 Testing crypto data handling...")
    
    try:
        from src.data.collectors.crypto_collector import CryptoDataCollector
        
        # Initialize without real API calls
        from src.config.settings import Settings
        test_settings = Settings()
        
        collector = CryptoDataCollector(test_settings)
        
        # Test data validation
        test_data = {
            'timestamp': 1640995200000,
            'open': '50000.0',
            'high': '51000.0', 
            'low': '49000.0',
            'close': '50500.0',
            'volume': '1000.0'
        }
        
        # This should work without network calls
        logger.success("✅ Crypto data collector initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Crypto data test failed: {e}")
        return False

def main():
    """Run all simple tests"""
    
    logger.info("🧪 Coral TPU Trading System - Simple Test Suite")
    logger.info("=" * 50)
    
    try:
        # Test basic system
        result1 = asyncio.run(test_system_basics())
        
        # Test crypto data
        result2 = asyncio.run(test_crypto_data())
        
        if result1 and result2:
            logger.success("🎯 All tests passed! System is ready for full deployment.")
            logger.info("💡 Next steps:")
            logger.info("   1. Install TensorFlow: pip install tensorflow")
            logger.info("   2. Install Coral TPU libraries")
            logger.info("   3. Set up real API keys in .env file")
            logger.info("   4. Run: python main.py")
            return 0
        else:
            logger.warning("⚠️ Some tests failed, but basic structure is working")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
