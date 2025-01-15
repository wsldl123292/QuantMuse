import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

@dataclass
class ProcessedData:
    """Container for processed market data"""
    raw_data: pd.DataFrame
    indicators: Dict[str, pd.Series]
    statistics: Dict[str, float]
    signals: Dict[str, bool]

class DataProcessor:
    """Processes market data and calculates technical indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_market_data(self, df: pd.DataFrame) -> ProcessedData:
        """Process market data and calculate indicators"""
        try:
            indicators = {}
            
            # Calculate moving averages
            indicators['sma_20'] = self._calculate_sma(df['close'], 20)
            indicators['sma_50'] = self._calculate_sma(df['close'], 50)
            indicators['ema_12'] = self._calculate_ema(df['close'], 12)
            indicators['ema_26'] = self._calculate_ema(df['close'], 26)
            
            # Calculate momentum indicators
            indicators['rsi'] = self._calculate_rsi(df['close'])
            indicators['macd'], indicators['signal'] = self._calculate_macd(df['close'])
            
            # Calculate volatility indicators
            indicators['bollinger_upper'], indicators['bollinger_lower'] = \
                self._calculate_bollinger_bands(df['close'])
            
            # Calculate statistics
            statistics = self._calculate_statistics(df)
            
            # Generate trading signals
            signals = self._generate_signals(df, indicators)
            
            return ProcessedData(
                raw_data=df,
                indicators=indicators,
                statistics=statistics,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            raise

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD and Signal line"""
        fast_ema = self._calculate_ema(series, fast)
        slow_ema = self._calculate_ema(series, slow)
        macd = fast_ema - slow_ema
        signal_line = self._calculate_ema(macd, signal)
        return macd, signal_line

    def _calculate_bollinger_bands(self, series: pd.Series, 
                                 period: int = 20, std: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(series, period)
        std_dev = series.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market statistics"""
        return {
            'daily_return': df['close'].pct_change().mean(),
            'volatility': df['close'].pct_change().std() * np.sqrt(252),
            'current_price': df['close'].iloc[-1],
            'volume': df['volume'].mean(),
            'high_52w': df['high'].rolling(window=252).max().iloc[-1],
            'low_52w': df['low'].rolling(window=252).min().iloc[-1]
        }

    def _generate_signals(self, df: pd.DataFrame, 
                         indicators: Dict[str, pd.Series]) -> Dict[str, bool]:
        """Generate trading signals based on indicators"""
        return {
            'is_golden_cross': indicators['sma_20'].iloc[-1] > indicators['sma_50'].iloc[-1],
            'is_death_cross': indicators['sma_20'].iloc[-1] < indicators['sma_50'].iloc[-1],
            'is_overbought': indicators['rsi'].iloc[-1] > 70,
            'is_oversold': indicators['rsi'].iloc[-1] < 30,
            'is_macd_bullish': indicators['macd'].iloc[-1] > indicators['signal'].iloc[-1],
            'is_macd_bearish': indicators['macd'].iloc[-1] < indicators['signal'].iloc[-1]
        } 