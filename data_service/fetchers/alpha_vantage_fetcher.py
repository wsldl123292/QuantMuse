from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any

class AlphaVantageFetcher:
    """Fetches data from Alpha Vantage API"""
    
    def __init__(self, api_key: str):
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.logger = logging.getLogger(__name__)

    def fetch_historical_data(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """Fetch historical stock data"""
        try:
            if interval == 'daily':
                data, _ = self.ts.get_daily(symbol=symbol, outputsize='full')
            elif interval == 'intraday':
                data, _ = self.ts.get_intraday(symbol=symbol, interval='1min')
            else:
                raise ValueError(f"Unsupported interval: {interval}")

            # Rename columns to match our format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return data

        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data: {str(e)}")
            raise

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch company information"""
        try:
            overview = self.ts.get_company_overview(symbol=symbol)
            return overview
        except Exception as e:
            self.logger.error(f"Error fetching company info: {str(e)}")
            raise 