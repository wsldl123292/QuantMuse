from binance.client import Client
from binance.websockets import BinanceSocketManager
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Optional

class BinanceFetcher:
    """
    A class to handle all Binance API interactions for data fetching.
    
    This class provides methods to:
    - Fetch historical cryptocurrency data
    - Stream real-time market data
    - Get order book information
    
    Attributes:
        client: Binance API client instance
        bm: Binance WebSocket manager
        logger: Logger instance for tracking operations
        _ws_connection: WebSocket connection handler
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the Binance fetcher with API credentials.
        
        Args:
            api_key (str, optional): Binance API key
            api_secret (str, optional): Binance API secret key
            
        Raises:
            BinanceAPIException: If API credentials are invalid
        """
        self.client = Client(api_key, api_secret)
        self.bm = BinanceSocketManager(self.client)
        self.logger = logging.getLogger(__name__)
        self._ws_connection = None
        self._callbacks = []

    def fetch_historical_data(self, symbol: str, interval: str = '1d', 
                            limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical kline/candlestick data from Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Candlestick interval (e.g., '1d', '1h', '15m')
            limit (int): Number of candlesticks to fetch
            
        Returns:
            pd.DataFrame: DataFrame containing historical market data with columns:
                - timestamp: Datetime index
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume
                
        Raises:
            DataFetchException: If data fetching fails
        """
        try:
            # Fetch raw kline data from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame and process
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Process timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def fetch_realtime_data(self, symbol: str, callback):
        """
        Start websocket connection for real-time market data.
        
        Args:
            symbol (str): Trading pair symbol
            callback (callable): Function to handle incoming data
            
        The callback function will receive data in the format:
        {
            'symbol': str,
            'price': float,
            'quantity': float,
            'timestamp': datetime
        }
        """
        def handle_socket_message(msg):
            """Internal handler for websocket messages"""
            try:
                if msg.get('e') == 'trade':
                    # Process trade data
                    data = {
                        'symbol': msg['s'],
                        'price': float(msg['p']),
                        'quantity': float(msg['q']),
                        'timestamp': pd.to_datetime(msg['T'], unit='ms')
                    }
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error processing websocket message: {str(e)}")

        try:
            # Start websocket connection
            self._ws_connection = self.bm.start_trade_socket(
                symbol=symbol,
                callback=handle_socket_message
            )
            self.bm.start()
            self.logger.info(f"Started real-time data stream for {symbol}")
        except Exception as e:
            self.logger.error(f"Error starting websocket: {str(e)}")
            raise

    def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current order book data.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Order book data with structure:
                {
                    'bids': [[price, quantity], ...],
                    'asks': [[price, quantity], ...]
                }
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=100)
            return {
                'bids': [[float(price), float(qty)] for price, qty in depth['bids']],
                'asks': [[float(price), float(qty)] for price, qty in depth['asks']]
            }
        except Exception as e:
            self.logger.error(f"Error fetching market depth: {str(e)}")
            raise

    def stop_realtime_data(self):
        """
        Stop websocket connection and cleanup resources.
        Should be called when real-time data is no longer needed.
        """
        if self._ws_connection:
            self.bm.stop_socket(self._ws_connection)
            self.bm.close()
            self._ws_connection = None
            self.logger.info("Stopped real-time data stream") 