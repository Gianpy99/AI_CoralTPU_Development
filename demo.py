"""
Demo script showing how to use the Coral TPU Crypto Trading System
"""

import asyncio
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


async def demo_system():
    """Demonstrate the trading system with dummy data"""
    
    logger.info("🚀 Starting Coral TPU Crypto Trading System Demo")
    logger.info("=" * 60)
    
    try:
        # Import system components
        from src.config.settings import Settings
        from src.models.inference.tpu_inference import TPUInferenceEngine
        from src.trading.engine.trading_engine import TradingEngine
        from src.utils.monitoring.dashboard import DashboardServer
        
        # Initialize settings
        logger.info("📋 Initializing system settings...")
        settings = Settings()
        settings.TRADING_MODE = "simulation"  # Force simulation mode
        logger.info("✓ Settings loaded")
        
        # Initialize TPU inference engine
        logger.info("🧠 Initializing AI inference engine...")
        model_path = "models/crypto_predictor.tflite"
        inference_engine = TPUInferenceEngine(model_path)
        logger.info("✓ Inference engine ready")
        
        # Initialize trading engine
        logger.info("💰 Initializing trading engine...")
        trading_engine = TradingEngine(settings, inference_engine)
        logger.info("✓ Trading engine ready")
        
        # Start dashboard (optional)
        dashboard = None
        try:
            logger.info("📊 Starting web dashboard...")
            dashboard = DashboardServer(port=8000)
            await dashboard.start()
            logger.info("✓ Dashboard available at http://localhost:8000")
        except Exception as e:
            logger.warning(f"Dashboard not available: {e}")
        
        # Demo loop
        logger.info("\n🎯 Starting trading demo...")
        logger.info("The system will simulate trading with dummy market data")
        logger.info("Press Ctrl+C to stop the demo")
        
        demo_duration = 60  # 1 minute demo
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < demo_duration:
            iteration += 1
            logger.info(f"\n--- Demo Iteration {iteration} ---")
            
            # Generate dummy market data
            market_data = generate_dummy_market_data()
            logger.info(f"📈 Generated market data for {len(market_data)} symbols")
            
            # Process market data and generate signals
            signals = await trading_engine.process_market_data(market_data)
            
            if signals:
                logger.info(f"📊 Generated {len(signals)} trading signals:")
                for signal in signals:
                    logger.info(f"  • {signal.symbol}: {signal.signal_type.value.upper()} "
                               f"(confidence: {signal.confidence:.2f})")
                
                # Execute trades
                await trading_engine.execute_trades(signals)
            else:
                logger.info("📊 No trading signals generated")
            
            # Update portfolio
            await trading_engine.update_positions(market_data)
            
            # Show portfolio summary
            summary = trading_engine.get_portfolio_summary()
            logger.info(f"💼 Portfolio Summary:")
            logger.info(f"  • Total Value: ${summary['total_portfolio_value']:.2f}")
            logger.info(f"  • P&L: ${summary['total_pnl']:.2f}")
            logger.info(f"  • Active Positions: {summary['active_positions']}")
            logger.info(f"  • Win Rate: {summary['win_rate']:.1%}")
            
            # Update dashboard
            if dashboard:
                dashboard.update_stats(summary)
            
            # Wait before next iteration
            await asyncio.sleep(5)
        
        logger.info("\n🏁 Demo completed!")
        
        # Final summary
        final_summary = trading_engine.get_portfolio_summary()
        logger.info("\n📈 FINAL RESULTS:")
        logger.info(f"  • Starting Value: $10,000.00")
        logger.info(f"  • Final Value: ${final_summary['total_portfolio_value']:.2f}")
        logger.info(f"  • Total Return: {final_summary['total_return']:.2%}")
        logger.info(f"  • Total P&L: ${final_summary['total_pnl']:.2f}")
        logger.info(f"  • Signals Generated: {final_summary['signals_generated']}")
        logger.info(f"  • Successful Trades: {final_summary['successful_trades']}")
        logger.info(f"  • Failed Trades: {final_summary['failed_trades']}")
        logger.info(f"  • Win Rate: {final_summary['win_rate']:.1%}")
        
        # Close all positions
        await trading_engine.close_all_positions()
        
        # Stop dashboard
        if dashboard:
            await dashboard.stop()
        
        logger.info("✨ Demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo stopped by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


def generate_dummy_market_data():
    """Generate realistic dummy market data for demo"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
    market_data = {}
    
    for symbol in symbols:
        # Generate 100 periods of dummy OHLCV data
        periods = 100
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=periods),
            periods=periods,
            freq='1H'
        )
        
        # Generate realistic price movements
        base_price = {
            "BTCUSDT": 45000,
            "ETHUSDT": 3000,
            "ADAUSDT": 0.5,
            "SOLUSDT": 100
        }[symbol]
        
        # Random walk with trend
        returns = np.random.normal(0.0001, 0.02, periods)  # Small positive trend with volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))  # Prevent negative prices
        
        # Create OHLCV data
        opens = prices[:-1]
        closes = prices[1:]
        
        highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.01))) for o, c in zip(opens, closes)]
        lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.01))) for o, c in zip(opens, closes)]
        volumes = np.random.uniform(1000, 10000, periods-1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates[1:])
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        market_data[symbol] = df
    
    return market_data


def add_technical_indicators(df):
    """Add technical indicators to market data"""
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price momentum and volatility
    df['price_momentum'] = df['close'].pct_change(periods=5)
    df['volatility'] = df['close'].rolling(window=20).std()
    
    return df


async def quick_test():
    """Quick test of core functionality"""
    logger.info("🔧 Running quick system test...")
    
    try:
        # Test settings
        from src.config.settings import Settings
        settings = Settings()
        logger.info("✓ Settings loaded")
        
        # Test inference engine
        from src.models.inference.tpu_inference import TPUInferenceEngine
        engine = TPUInferenceEngine("models/dummy_model.tflite")
        logger.info("✓ Inference engine created")
        
        # Test with dummy data
        dummy_data = generate_dummy_market_data()["BTCUSDT"]
        processed = engine.preprocess_data(dummy_data)
        prediction = engine.predict(processed)
        logger.info(f"✓ Prediction: {prediction}")
        
        logger.info("🎉 Quick test passed!")
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        raise


def show_system_info():
    """Show system information and requirements"""
    
    logger.info("📋 CORAL TPU CRYPTO TRADING SYSTEM")
    logger.info("=" * 50)
    
    # System info
    import platform
    import sys
    
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    
    # Check dependencies
    logger.info("\n🔍 Checking Dependencies:")
    
    dependencies = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("tensorflow", "TensorFlow"),
        ("pycoral", "PyCoral"),
        ("tflite_runtime", "TFLite Runtime"),
        ("ccxt", "CCXT"),
        ("loguru", "Loguru"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.warning(f"⚠ {name} - Not installed")
    
    # Project structure
    logger.info("\n📁 Project Structure:")
    project_root = Path(__file__).parent
    
    important_paths = [
        "src/config/settings.py",
        "src/data/collectors/crypto_collector.py",
        "src/models/inference/tpu_inference.py",
        "src/trading/engine/trading_engine.py",
        "src/utils/monitoring/dashboard.py",
        "requirements.txt",
        ".env.template"
    ]
    
    for path in important_paths:
        full_path = project_root / path
        if full_path.exists():
            logger.info(f"✓ {path}")
        else:
            logger.warning(f"⚠ {path} - Missing")
    
    logger.info("\n📚 Getting Started:")
    logger.info("1. Install dependencies: pip install -r requirements.txt")
    logger.info("2. Copy .env.template to .env and configure API keys")
    logger.info("3. Test system: python demo.py --test")
    logger.info("4. Run demo: python demo.py")
    logger.info("5. Start full system: python main.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coral TPU Crypto Trading System Demo")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--info", action="store_true", help="Show system information")
    
    args = parser.parse_args()
    
    if args.info:
        show_system_info()
    elif args.test:
        asyncio.run(quick_test())
    else:
        asyncio.run(demo_system())
