import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
from data_service.fetchers.binance_fetcher import BinanceFetcher
from data_service.utils.exceptions import DataFetchError

class TestBinanceFetcher(unittest.TestCase):
    """Test cases for BinanceFetcher class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.fetcher = BinanceFetcher()
        
    def test_initialization(self):
        """Test fetcher initialization"""
        self.assertIsNotNone(self.fetcher.client)
        self.assertIsNotNone(self.fetcher.logger)
        
    @patch('binance.client.Client.get_historical_klines')
    def test_fetch_historical_data_success(self, mock_get_klines):
        """Test successful historical data fetch"""
        # Mock data
        mock_data = [
            [
                1499040000000,      # Timestamp
                "8100.0",           # Open
                "8200.0",           # High
                "8000.0",           # Low
                "8150.0",           # Close
                "100.0",            # Volume
                1499644799999,      # Close time
                "1000.0",           # Quote volume
                100,                # Number of trades
                "50.0",             # Taker buy base
                "400.0",            # Taker buy quote
                "0"                 # Ignore
            ]
        ]
        
        mock_get_klines.return_value = mock_data
        
        # Test
        df = self.fetcher.fetch_historical_data("BTCUSDT")
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['close'].iloc[0], 8150.0)
        
    @patch('binance.client.Client.get_historical_klines')
    def test_fetch_historical_data_error(self, mock_get_klines):
        """Test error handling in historical data fetch"""
        mock_get_klines.side_effect = Exception("API Error")
        
        with self.assertRaises(DataFetchError):
            self.fetcher.fetch_historical_data("BTCUSDT")
            
    def test_invalid_symbol(self):
        """Test handling of invalid symbol"""
        with self.assertRaises(ValidationError):
            self.fetcher.fetch_historical_data("") 