import unittest
import pandas as pd
import numpy as np
from data_service.processors.data_processor import DataProcessor
from data_service.utils.exceptions import ProcessingError

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.processor = DataProcessor()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.sample_data = pd.DataFrame({
            'open': np.random.randn(len(dates)) + 100,
            'high': np.random.randn(len(dates)) + 101,
            'low': np.random.randn(len(dates)) + 99,
            'close': np.random.randn(len(dates)) + 100,
            'volume': np.random.randn(len(dates)) * 1000
        }, index=dates)
        
    def test_process_market_data(self):
        """Test market data processing"""
        result = self.processor.process_market_data(self.sample_data)
        
        # Check structure
        self.assertIsNotNone(result.indicators)
        self.assertIsNotNone(result.statistics)
        self.assertIsNotNone(result.signals)
        
        # Check indicators
        self.assertIn('sma_20', result.indicators)
        self.assertIn('rsi', result.indicators)
        self.assertIn('macd', result.indicators)
        
        # Check statistics
        self.assertIn('daily_return', result.statistics)
        self.assertIn('volatility', result.statistics)
        
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        rsi = self.processor._calculate_rsi(self.sample_data['close'])
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))
        
    def test_error_handling(self):
        """Test error handling with invalid data"""
        invalid_data = pd.DataFrame()
        with self.assertRaises(ProcessingError):
            self.processor.process_market_data(invalid_data) 