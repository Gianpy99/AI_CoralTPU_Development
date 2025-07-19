"""
Cryptocurrency data collector using multiple exchange APIs
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import ccxt
from loguru import logger
import numpy as np


class CryptoDataCollector:
    """Collects real-time and historical cryptocurrency data from multiple exchanges"""
    
    def __init__(self, settings):
        self.settings = settings
        self.exchanges = {}
        self.symbols = settings.DEFAULT_SYMBOLS
        self.data_cache = {}
        self.last_update = {}
        
    async def initialize(self):
        """Initialize exchange connections"""
        logger.info("Initializing crypto data collector...")
        
        try:
            # Initialize Binance
            if self.settings.BINANCE_API_KEY:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.settings.BINANCE_API_KEY,
                    'secret': self.settings.BINANCE_SECRET_KEY,
                    'sandbox': self.settings.TRADING_MODE != 'live',
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
            
            # Initialize other exchanges as needed
            # self.exchanges['coinbase'] = ccxt.coinbasepro({...})
            
            # Test connections
            for exchange_name, exchange in self.exchanges.items():
                try:
                    markets = await self._fetch_markets(exchange)
                    logger.success(f"Connected to {exchange_name}: {len(markets)} markets available")
                except Exception as e:
                    logger.error(f"Failed to connect to {exchange_name}: {e}")
                    
            logger.success("Data collector initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize data collector: {e}")
            raise
    
    async def _fetch_markets(self, exchange) -> Dict:
        """Fetch available markets from exchange"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, exchange.load_markets)
    
    async def get_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Get latest market data for all symbols"""
        data = {}
        
        for symbol in self.symbols:
            try:
                # Get data from primary exchange (Binance)
                if 'binance' in self.exchanges:
                    symbol_data = await self._get_symbol_data(symbol, 'binance')
                    if symbol_data is not None:
                        data[symbol] = symbol_data
                        
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    async def _get_symbol_data(self, symbol: str, exchange_name: str) -> Optional[pd.DataFrame]:
        """Get comprehensive data for a single symbol"""
        exchange = self.exchanges[exchange_name]
        
        try:
            # Get OHLCV data
            ohlcv_data = await self._fetch_ohlcv(exchange, symbol, '1m', limit=100)
            
            # Get order book
            orderbook = await self._fetch_orderbook(exchange, symbol)
            
            # Get recent trades
            trades = await self._fetch_trades(exchange, symbol, limit=50)
            
            # Get ticker
            ticker = await self._fetch_ticker(exchange, symbol)
            
            # Combine all data into a comprehensive DataFrame
            df = self._create_market_dataframe(ohlcv_data, orderbook, trades, ticker)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Cache the data
            self.data_cache[symbol] = df
            self.last_update[symbol] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} data from {exchange_name}: {e}")
            return None
    
    async def _fetch_ohlcv(self, exchange, symbol: str, timeframe: str, limit: int = 100) -> List:
        """Fetch OHLCV data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            exchange.fetch_ohlcv, 
            symbol, 
            timeframe, 
            None, 
            limit
        )
    
    async def _fetch_orderbook(self, exchange, symbol: str) -> Dict:
        """Fetch order book data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, exchange.fetch_order_book, symbol)
    
    async def _fetch_trades(self, exchange, symbol: str, limit: int = 50) -> List:
        """Fetch recent trades"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, exchange.fetch_trades, symbol, None, limit)
    
    async def _fetch_ticker(self, exchange, symbol: str) -> Dict:
        """Fetch ticker data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, exchange.fetch_ticker, symbol)
    
    def _create_market_dataframe(self, ohlcv: List, orderbook: Dict, 
                                trades: List, ticker: Dict) -> pd.DataFrame:
        """Create a comprehensive market data DataFrame"""
        # Convert OHLCV to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Add current ticker info to the latest row
        if not df.empty:
            latest_idx = df.index[-1]
            
            # Order book metrics
            df.loc[latest_idx, 'bid_price'] = orderbook['bids'][0][0] if orderbook['bids'] else np.nan
            df.loc[latest_idx, 'ask_price'] = orderbook['asks'][0][0] if orderbook['asks'] else np.nan
            df.loc[latest_idx, 'bid_volume'] = orderbook['bids'][0][1] if orderbook['bids'] else np.nan
            df.loc[latest_idx, 'ask_volume'] = orderbook['asks'][0][1] if orderbook['asks'] else np.nan
            df.loc[latest_idx, 'spread'] = (df.loc[latest_idx, 'ask_price'] - df.loc[latest_idx, 'bid_price'])
            
            # Calculate order book depth
            total_bid_volume = sum([bid[1] for bid in orderbook['bids'][:10]])  # Top 10 bids
            total_ask_volume = sum([ask[1] for ask in orderbook['asks'][:10]])  # Top 10 asks
            df.loc[latest_idx, 'bid_depth'] = total_bid_volume
            df.loc[latest_idx, 'ask_depth'] = total_ask_volume
            
            # Trade metrics
            if trades:
                recent_trade_volume = sum([trade['amount'] for trade in trades[-10:]])
                recent_trade_price_avg = np.mean([trade['price'] for trade in trades[-10:]])
                df.loc[latest_idx, 'recent_trade_volume'] = recent_trade_volume
                df.loc[latest_idx, 'recent_trade_price_avg'] = recent_trade_price_avg
            
            # Ticker metrics
            df.loc[latest_idx, 'price_change_24h'] = ticker.get('change', 0)
            df.loc[latest_idx, 'price_change_percent_24h'] = ticker.get('percentage', 0)
            df.loc[latest_idx, 'volume_24h'] = ticker.get('baseVolume', 0)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame"""
        if len(df) < 20:  # Need enough data for indicators
            return df
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean() if len(df) >= 50 else np.nan
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
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
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(periods=5)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        return df
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                                 days: int = 30) -> pd.DataFrame:
        """Get historical data for training models"""
        if 'binance' not in self.exchanges:
            raise ValueError("No exchange available for historical data")
        
        exchange = self.exchanges['binance']
        
        # Calculate the number of candles needed
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes_per_day = 24 * 60
        total_minutes = days * minutes_per_day
        candles_needed = total_minutes // timeframe_minutes[timeframe]
        
        # Fetch data in chunks (max 1000 per request)
        all_data = []
        chunk_size = 1000
        
        for i in range(0, candles_needed, chunk_size):
            limit = min(chunk_size, candles_needed - i)
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000) + (i * timeframe_minutes[timeframe] * 60 * 1000)
            
            try:
                chunk_data = await self._fetch_ohlcv(exchange, symbol, timeframe, limit)
                all_data.extend(chunk_data)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error fetching historical chunk: {e}")
                break
        
        if not all_data:
            raise ValueError(f"No historical data available for {symbol}")
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
    async def close(self):
        """Close all exchange connections"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                if hasattr(exchange, 'close'):
                    await exchange.close()
                logger.info(f"Closed connection to {exchange_name}")
            except Exception as e:
                logger.error(f"Error closing {exchange_name}: {e}")


# Standalone script for data collection
if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append("../..")
    from src.config.settings import Settings
    
    async def main():
        parser = argparse.ArgumentParser(description="Collect crypto data")
        parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to collect")
        parser.add_argument("--hours", type=int, default=24, help="Hours of historical data")
        parser.add_argument("--output", help="Output CSV file")
        
        args = parser.parse_args()
        
        settings = Settings()
        collector = CryptoDataCollector(settings)
        
        try:
            await collector.initialize()
            
            # Get historical data
            df = await collector.get_historical_data(
                symbol=args.symbol,
                timeframe='1h',
                days=args.hours // 24
            )
            
            # Save to file
            output_file = args.output or f"data/{args.symbol}_{args.hours}h.csv"
            df.to_csv(output_file)
            
            logger.success(f"Data saved to {output_file}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
        finally:
            await collector.close()
    
    asyncio.run(main())
