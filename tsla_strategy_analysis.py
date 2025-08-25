#!/usr/bin/env python3
"""
TSLA (特斯拉) 完整策略分析脚本
Author: LDL
Date: 2025-01-25

这个脚本演示如何使用QuantMuse系统对TSLA进行全面的量化分析，包括：
1. 数据获取
2. 因子计算
3. 策略回测
4. AI分析
5. 可视化展示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入QuantMuse模块
from data_service.fetchers import YahooFetcher
from data_service.factors import FactorCalculator, FactorScreener, FactorBacktest
from data_service.strategies import MomentumStrategy, StrategyRunner
from data_service.backtest import BacktestEngine
from data_service.processors import DataProcessor
from data_service.utils.logger import setup_logger

# 尝试导入AI模块（如果配置了API密钥）
try:
    from data_service.ai import LLMIntegration, SentimentAnalyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  AI模块未配置，将跳过AI分析部分")

class TSLAStrategyAnalyzer:
    """TSLA策略分析器"""
    
    def __init__(self):
        self.logger = setup_logger("tsla_analyzer")
        self.symbol = "TSLA"
        
        # 初始化组件
        self.yahoo_fetcher = YahooFetcher()
        self.factor_calculator = FactorCalculator()
        self.factor_screener = FactorScreener()
        self.factor_backtest = FactorBacktest()
        self.data_processor = DataProcessor()
        self.backtest_engine = BacktestEngine(initial_capital=100000)
        
        # 数据存储
        self.price_data = None
        self.company_info = None
        self.factor_data = None
        
        self.logger.info(f"🚗 TSLA策略分析器初始化完成")
    
    def fetch_tsla_data(self, period: str = "2y"):
        """获取TSLA数据"""
        self.logger.info(f"📊 正在获取TSLA {period}期间的数据...")
        
        try:
            # 获取历史价格数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=730 if period == "2y" else 365)
            
            self.price_data = self.yahoo_fetcher.fetch_historical_data(
                symbol=self.symbol,
                start_time=start_time,
                end_time=end_time,
                interval='1d'
            )
            
            # 获取公司信息
            self.company_info = self.yahoo_fetcher.get_company_info(self.symbol)
            
            self.logger.info(f"✅ 成功获取 {len(self.price_data)} 条TSLA数据记录")
            self.logger.info(f"📈 数据时间范围: {self.price_data.index[0].date()} 到 {self.price_data.index[-1].date()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 获取TSLA数据失败: {str(e)}")
            return False
    
    def analyze_company_fundamentals(self):
        """分析公司基本面"""
        if not self.company_info:
            self.logger.warning("⚠️  公司信息未获取，跳过基本面分析")
            return
        
        self.logger.info("🏢 TSLA公司基本面分析:")
        self.logger.info(f"  公司名称: {self.company_info.get('name', 'N/A')}")
        self.logger.info(f"  行业: {self.company_info.get('industry', 'N/A')}")
        self.logger.info(f"  市值: ${self.company_info.get('market_cap', 0):,.0f}")
        self.logger.info(f"  P/E比率: {self.company_info.get('pe_ratio', 'N/A')}")
        self.logger.info(f"  Beta系数: {self.company_info.get('beta', 'N/A')}")
        self.logger.info(f"  股息收益率: {self.company_info.get('dividend_yield', 'N/A')}")
    
    def calculate_factors(self):
        """计算TSLA的量化因子"""
        if self.price_data is None:
            self.logger.error("❌ 价格数据未获取，无法计算因子")
            return False
        
        self.logger.info("🧮 正在计算TSLA量化因子...")
        
        try:
            # 计算所有因子
            factors = self.factor_calculator.calculate_all_factors(
                symbol=self.symbol,
                prices=self.price_data['close'],
                volumes=self.price_data['volume']
            )
            
            # 转换为DataFrame格式
            self.factor_data = pd.DataFrame([factors])
            self.factor_data['symbol'] = self.symbol
            self.factor_data['date'] = self.price_data.index[-1]
            
            self.logger.info("✅ 因子计算完成，主要因子值:")
            for factor_name, value in factors.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {factor_name}: {value:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 因子计算失败: {str(e)}")
            return False
    
    def run_momentum_strategy(self):
        """运行动量策略"""
        if self.price_data is None:
            self.logger.error("❌ 价格数据未获取，无法运行策略")
            return None
        
        self.logger.info("🎯 正在运行TSLA动量策略...")
        
        try:
            # 创建动量策略
            momentum_strategy = MomentumStrategy()
            
            # 运行回测
            results = self.backtest_engine.run_backtest(
                strategy=momentum_strategy,
                data=self.price_data
            )
            
            self.logger.info("✅ 动量策略回测完成:")
            self.logger.info(f"  总收益率: {results.total_return:.2%}")
            self.logger.info(f"  年化收益率: {results.annualized_return:.2%}")
            self.logger.info(f"  夏普比率: {results.sharpe_ratio:.3f}")
            self.logger.info(f"  最大回撤: {results.max_drawdown:.2%}")
            self.logger.info(f"  胜率: {results.win_rate:.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 动量策略运行失败: {str(e)}")
            return None
    
    def technical_analysis(self):
        """技术分析"""
        if self.price_data is None:
            return
        
        self.logger.info("📊 正在进行技术分析...")
        
        # 处理数据获取技术指标
        processed_data = self.data_processor.process_market_data(self.price_data)
        
        self.logger.info("📈 技术分析结果:")
        for key, value in processed_data.statistics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        self.logger.info("🚦 交易信号:")
        for key, value in processed_data.signals.items():
            self.logger.info(f"  {key}: {value}")
    
    def ai_analysis(self):
        """AI分析（如果可用）"""
        if not AI_AVAILABLE:
            self.logger.info("⚠️  AI模块未配置，跳过AI分析")
            return
        
        self.logger.info("🤖 正在进行AI分析...")
        
        try:
            # 这里可以添加AI分析代码
            # llm = LLMIntegration(provider="openai")
            # analysis = llm.analyze_market(self.factor_data, self.price_data)
            self.logger.info("🤖 AI分析功能需要配置OpenAI API密钥")
            
        except Exception as e:
            self.logger.error(f"❌ AI分析失败: {str(e)}")
    
    def create_visualization(self):
        """创建可视化图表"""
        if self.price_data is None:
            return
        
        self.logger.info("📊 正在创建可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TSLA 策略分析报告', fontsize=16, fontweight='bold')
        
        # 1. 价格走势图
        axes[0, 0].plot(self.price_data.index, self.price_data['close'], linewidth=2)
        axes[0, 0].set_title('TSLA 价格走势')
        axes[0, 0].set_ylabel('价格 ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成交量
        axes[0, 1].bar(self.price_data.index, self.price_data['volume'], alpha=0.7)
        axes[0, 1].set_title('TSLA 成交量')
        axes[0, 1].set_ylabel('成交量')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 收益率分布
        returns = self.price_data['close'].pct_change().dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('日收益率分布')
        axes[1, 0].set_xlabel('收益率')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 移动平均线
        ma20 = self.price_data['close'].rolling(20).mean()
        ma50 = self.price_data['close'].rolling(50).mean()
        
        axes[1, 1].plot(self.price_data.index, self.price_data['close'], label='收盘价', linewidth=2)
        axes[1, 1].plot(self.price_data.index, ma20, label='MA20', alpha=0.8)
        axes[1, 1].plot(self.price_data.index, ma50, label='MA50', alpha=0.8)
        axes[1, 1].set_title('TSLA 移动平均线')
        axes[1, 1].set_ylabel('价格 ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"tsla_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 图表已保存为: {chart_filename}")
        
        plt.show()
    
    def generate_report(self):
        """生成分析报告"""
        self.logger.info("📝 正在生成TSLA策略分析报告...")
        
        report = f"""
        
🚗 TSLA (特斯拉) 量化策略分析报告
{'='*50}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析师: LDL

📊 数据概览:
- 分析标的: {self.symbol}
- 数据条数: {len(self.price_data) if self.price_data is not None else 'N/A'}
- 数据时间范围: {self.price_data.index[0].date() if self.price_data is not None else 'N/A'} 到 {self.price_data.index[-1].date() if self.price_data is not None else 'N/A'}
- 当前价格: ${self.price_data['close'].iloc[-1]:.2f} if self.price_data is not None else 'N/A'

🏢 公司基本面:
- 公司名称: {self.company_info.get('name', 'N/A') if self.company_info else 'N/A'}
- 行业: {self.company_info.get('industry', 'N/A') if self.company_info else 'N/A'}
- 市值: ${self.company_info.get('market_cap', 0):,.0f} if self.company_info else 'N/A'
- P/E比率: {self.company_info.get('pe_ratio', 'N/A') if self.company_info else 'N/A'}

📈 技术指标:
- 20日移动平均: ${self.price_data['close'].rolling(20).mean().iloc[-1]:.2f} if self.price_data is not None else 'N/A'
- 50日移动平均: ${self.price_data['close'].rolling(50).mean().iloc[-1]:.2f} if self.price_data is not None else 'N/A'
- 日均成交量: {self.price_data['volume'].mean():,.0f} if self.price_data is not None else 'N/A'

⚠️  风险提示:
本分析仅供学习研究使用，不构成投资建议。
投资有风险，入市需谨慎。

{'='*50}
        """
        
        print(report)
        
        # 保存报告到文件
        report_filename = f"tsla_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"📝 报告已保存为: {report_filename}")

def main():
    """主函数"""
    print("🚗 欢迎使用TSLA量化策略分析系统!")
    print("Author: LDL")
    print("="*50)
    
    # 创建分析器
    analyzer = TSLAStrategyAnalyzer()
    
    try:
        # 1. 获取数据
        if not analyzer.fetch_tsla_data(period="2y"):
            print("❌ 数据获取失败，程序退出")
            return
        
        # 2. 基本面分析
        analyzer.analyze_company_fundamentals()
        
        # 3. 计算因子
        analyzer.calculate_factors()
        
        # 4. 技术分析
        analyzer.technical_analysis()
        
        # 5. 运行策略
        analyzer.run_momentum_strategy()
        
        # 6. AI分析
        analyzer.ai_analysis()
        
        # 7. 创建可视化
        analyzer.create_visualization()
        
        # 8. 生成报告
        analyzer.generate_report()
        
        print("\n🎉 TSLA策略分析完成!")
        print("📊 请查看生成的图表和报告文件")
        
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
