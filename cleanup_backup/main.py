"""
Main entry point for the Coral TPU Crypto AI Trading System
"""

import asyncio
import signal
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import Settings
from src.data.collectors.crypto_collector import CryptoDataCollector
from src.models.inference.tpu_inference import TPUInferenceEngine
from src.trading.engine.trading_engine import TradingEngine
from src.utils.monitoring.dashboard import DashboardServer


class CoralTPUTradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self):
        self.settings = Settings()
        self.data_collector = None
        self.inference_engine = None
        self.trading_engine = None
        self.dashboard_server = None
        self.running = False
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing Coral TPU Trading System...")
        
        try:
            # Initialize data collector
            self.data_collector = CryptoDataCollector(self.settings)
            await self.data_collector.initialize()
            
            # Initialize TPU inference engine
            self.inference_engine = TPUInferenceEngine(
                model_path=self.settings.MODEL_PATH
            )
            
            # Initialize trading engine
            self.trading_engine = TradingEngine(
                settings=self.settings,
                inference_engine=self.inference_engine
            )
            
            # Initialize dashboard
            if self.settings.ENABLE_DASHBOARD:
                self.dashboard_server = DashboardServer(
                    port=self.settings.DASHBOARD_PORT
                )
                await self.dashboard_server.start()
            
            logger.success("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def run(self):
        """Main system loop"""
        self.running = True
        logger.info("Starting trading system main loop...")
        
        try:
            while self.running:
                # Collect latest market data
                market_data = await self.data_collector.get_latest_data()
                
                # Generate trading signals
                signals = await self.trading_engine.process_market_data(market_data)
                
                # Execute trades if signals are generated
                if signals:
                    await self.trading_engine.execute_trades(signals)
                
                # Wait for next iteration
                await asyncio.sleep(self.settings.LOOP_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading system...")
        self.running = False
        
        if self.trading_engine:
            await self.trading_engine.close_all_positions()
        
        if self.data_collector:
            await self.data_collector.close()
        
        if self.dashboard_server:
            await self.dashboard_server.stop()
        
        logger.success("System shutdown completed")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}")
    sys.exit(0)


async def main():
    """Main function"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run trading system
    system = CoralTPUTradingSystem()
    
    try:
        await system.initialize()
        await system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/trading_system.log",
        rotation="100 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    # Run the system
    asyncio.run(main())
