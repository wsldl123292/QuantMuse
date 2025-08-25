#!/usr/bin/env python3
"""
TSLA 快速分析脚本
Author: LDL
Date: 2025-01-25

快速运行TSLA策略分析的简化版本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def quick_tsla_analysis():
    """快速TSLA分析"""
    print("🚗 TSLA快速分析开始...")
    print("Author: LDL")
    print("="*50)
    
    try:
        # 导入必要模块
        from data_service.fetchers import YahooFetcher
        from data_service.factors import FactorCalculator
        from data_service.processors import DataProcessor
        
        # 1. 获取TSLA数据
        print("📊 正在获取TSLA数据...")
        fetcher = YahooFetcher()
        
        # 获取最近1年的数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        
        price_data = fetcher.fetch_historical_data(
            symbol="TSLA",
            start_time=start_time,
            end_time=end_time,
            interval='1d'
        )
        
        print(f"✅ 成功获取 {len(price_data)} 条数据记录")
        print(f"📈 时间范围: {price_data.index[0].date()} 到 {price_data.index[-1].date()}")
        
        # 2. 基本统计
        current_price = price_data['close'].iloc[-1]
        price_change = (current_price - price_data['close'].iloc[0]) / price_data['close'].iloc[0]
        
        print(f"\n💰 TSLA价格信息:")
        print(f"  当前价格: ${current_price:.2f}")
        print(f"  年度涨跌幅: {price_change:.2%}")
        print(f"  最高价: ${price_data['high'].max():.2f}")
        print(f"  最低价: ${price_data['low'].min():.2f}")
        
        # 3. 技术指标
        print(f"\n📊 技术指标:")
        ma20 = price_data['close'].rolling(20).mean().iloc[-1]
        ma50 = price_data['close'].rolling(50).mean().iloc[-1]
        
        print(f"  20日均线: ${ma20:.2f}")
        print(f"  50日均线: ${ma50:.2f}")
        print(f"  相对20日均线: {((current_price - ma20) / ma20):.2%}")
        print(f"  相对50日均线: {((current_price - ma50) / ma50):.2%}")
        
        # 4. 波动率分析
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        print(f"\n📈 风险指标:")
        print(f"  年化波动率: {volatility:.2%}")
        print(f"  最大单日涨幅: {returns.max():.2%}")
        print(f"  最大单日跌幅: {returns.min():.2%}")
        
        # 5. 简单因子计算
        try:
            factor_calc = FactorCalculator()
            factors = factor_calc.calculate_all_factors(
                symbol="TSLA",
                prices=price_data['close'],
                volumes=price_data['volume']
            )
            
            print(f"\n🧮 量化因子:")
            key_factors = ['momentum_20d', 'volatility', 'rsi', 'price_vs_ma20']
            for factor in key_factors:
                if factor in factors:
                    print(f"  {factor}: {factors[factor]:.4f}")
                    
        except Exception as e:
            print(f"⚠️  因子计算跳过: {str(e)}")
        
        # 6. 简单交易信号
        print(f"\n🚦 交易信号:")
        
        # 均线信号
        if current_price > ma20 > ma50:
            ma_signal = "看涨 📈"
        elif current_price < ma20 < ma50:
            ma_signal = "看跌 📉"
        else:
            ma_signal = "震荡 ↔️"
        
        print(f"  均线信号: {ma_signal}")
        
        # 动量信号
        momentum_5d = (current_price - price_data['close'].iloc[-6]) / price_data['close'].iloc[-6]
        if momentum_5d > 0.02:
            momentum_signal = "强势 🚀"
        elif momentum_5d < -0.02:
            momentum_signal = "弱势 📉"
        else:
            momentum_signal = "平稳 ➡️"
        
        print(f"  5日动量: {momentum_signal} ({momentum_5d:.2%})")
        
        # 7. 成交量分析
        avg_volume = price_data['volume'].rolling(20).mean().iloc[-1]
        current_volume = price_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        print(f"\n📊 成交量分析:")
        print(f"  最新成交量: {current_volume:,.0f}")
        print(f"  20日平均量: {avg_volume:,.0f}")
        print(f"  量比: {volume_ratio:.2f}")
        
        # 8. 简单建议
        print(f"\n💡 快速建议:")
        
        signals = []
        if current_price > ma20:
            signals.append("价格在20日均线上方")
        if momentum_5d > 0:
            signals.append("短期动量向上")
        if volume_ratio > 1.2:
            signals.append("成交量放大")
        
        if len(signals) >= 2:
            suggestion = "偏向积极 ✅"
        elif len(signals) == 1:
            suggestion = "谨慎观望 ⚠️"
        else:
            suggestion = "偏向谨慎 ❌"
        
        print(f"  综合评价: {suggestion}")
        print(f"  支持信号: {', '.join(signals) if signals else '无明显信号'}")
        
        print(f"\n⚠️  风险提示:")
        print(f"  本分析仅供参考，不构成投资建议")
        print(f"  投资有风险，决策需谨慎")
        
        print(f"\n🎉 TSLA快速分析完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = quick_tsla_analysis()
    
    if success:
        print(f"\n📝 如需详细分析，请运行:")
        print(f"  python tsla_strategy_analysis.py")
        print(f"\n📊 如需Web界面，请运行:")
        print(f"  python run_dashboard.py")
    
    input(f"\n按Enter键退出...")

if __name__ == "__main__":
    main()
