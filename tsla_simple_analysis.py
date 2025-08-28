#!/usr/bin/env python3
"""
TSLA 简化分析脚本
Author: LDL
Date: 2025-01-25

直接使用yfinance进行TSLA分析，避免复杂依赖
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_tsla():
    """TSLA简化分析"""
    print("🚗 TSLA简化分析开始...")
    print("Author: LDL")
    print("="*50)
    
    try:
        # 1. 获取TSLA数据
        print("📊 正在获取TSLA数据...")
        
        # 创建ticker对象
        tsla = yf.Ticker("TSLA")
        
        # 获取最近1年的数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        
        # 获取历史数据
        hist_data = tsla.history(start=start_time, end=end_time)
        
        if hist_data.empty:
            print("❌ 无法获取TSLA数据")
            return False
        
        print(f"✅ 成功获取 {len(hist_data)} 条数据记录")
        print(f"📈 时间范围: {hist_data.index[0].date()} 到 {hist_data.index[-1].date()}")
        
        # 2. 基本信息
        current_price = hist_data['Close'].iloc[-1]
        start_price = hist_data['Close'].iloc[0]
        price_change = (current_price - start_price) / start_price
        
        print(f"\n💰 TSLA价格信息:")
        print(f"  当前价格: ${current_price:.2f}")
        print(f"  年度涨跌幅: {price_change:.2%}")
        print(f"  最高价: ${hist_data['High'].max():.2f}")
        print(f"  最低价: ${hist_data['Low'].min():.2f}")
        
        # 3. 技术指标
        print(f"\n📊 技术指标:")
        
        # 移动平均线
        ma5 = hist_data['Close'].rolling(5).mean().iloc[-1]
        ma20 = hist_data['Close'].rolling(20).mean().iloc[-1]
        ma50 = hist_data['Close'].rolling(50).mean().iloc[-1]
        
        print(f"  5日均线: ${ma5:.2f}")
        print(f"  20日均线: ${ma20:.2f}")
        print(f"  50日均线: ${ma50:.2f}")
        print(f"  相对20日均线: {((current_price - ma20) / ma20):.2%}")
        print(f"  相对50日均线: {((current_price - ma50) / ma50):.2%}")
        
        # 4. 波动率分析
        returns = hist_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        print(f"\n📈 风险指标:")
        print(f"  年化波动率: {volatility:.2%}")
        print(f"  最大单日涨幅: {returns.max():.2%}")
        print(f"  最大单日跌幅: {returns.min():.2%}")
        
        # 5. RSI计算
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(hist_data['Close']).iloc[-1]
        
        print(f"\n🧮 量化指标:")
        print(f"  RSI(14): {rsi:.2f}")
        
        # RSI解读
        if rsi > 70:
            rsi_signal = "超买 ⚠️"
        elif rsi < 30:
            rsi_signal = "超卖 📈"
        else:
            rsi_signal = "正常 ➡️"
        print(f"  RSI信号: {rsi_signal}")
        
        # 6. 交易信号
        print(f"\n🚦 交易信号:")
        
        # 均线信号
        if current_price > ma20 > ma50:
            ma_signal = "看涨 📈"
        elif current_price < ma20 < ma50:
            ma_signal = "看跌 📉"
        else:
            ma_signal = "震荡 ↔️"
        
        print(f"  均线信号: {ma_signal}")
        
        # 短期动量
        momentum_5d = (current_price - hist_data['Close'].iloc[-6]) / hist_data['Close'].iloc[-6]
        if momentum_5d > 0.02:
            momentum_signal = "强势 🚀"
        elif momentum_5d < -0.02:
            momentum_signal = "弱势 📉"
        else:
            momentum_signal = "平稳 ➡️"
        
        print(f"  5日动量: {momentum_signal} ({momentum_5d:.2%})")
        
        # 7. 成交量分析
        avg_volume = hist_data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = hist_data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        print(f"\n📊 成交量分析:")
        print(f"  最新成交量: {current_volume:,.0f}")
        print(f"  20日平均量: {avg_volume:,.0f}")
        print(f"  量比: {volume_ratio:.2f}")
        
        # 8. 公司信息
        try:
            info = tsla.info
            print(f"\n🏢 公司信息:")
            print(f"  公司名称: {info.get('longName', 'Tesla, Inc.')}")
            print(f"  行业: {info.get('industry', 'Auto Manufacturers')}")
            print(f"  市值: ${info.get('marketCap', 0):,.0f}")
            print(f"  P/E比率: {info.get('trailingPE', 'N/A')}")
            print(f"  Beta系数: {info.get('beta', 'N/A')}")
        except:
            print(f"\n🏢 公司信息: 获取失败，跳过")
        
        # 9. 综合评价
        print(f"\n💡 综合评价:")
        
        signals = []
        if current_price > ma20:
            signals.append("价格在20日均线上方")
        if momentum_5d > 0:
            signals.append("短期动量向上")
        if volume_ratio > 1.2:
            signals.append("成交量放大")
        if 30 < rsi < 70:
            signals.append("RSI处于正常区间")
        
        if len(signals) >= 3:
            suggestion = "偏向积极 ✅"
        elif len(signals) >= 2:
            suggestion = "谨慎乐观 ⚠️"
        elif len(signals) >= 1:
            suggestion = "谨慎观望 ⚠️"
        else:
            suggestion = "偏向谨慎 ❌"
        
        print(f"  综合评价: {suggestion}")
        print(f"  支持信号: {', '.join(signals) if signals else '无明显信号'}")
        
        # 10. 风险提示
        print(f"\n⚠️  风险提示:")
        print(f"  • 本分析仅供参考，不构成投资建议")
        print(f"  • 投资有风险，决策需谨慎")
        print(f"  • 特斯拉股票波动性较大，注意风险控制")
        
        print(f"\n🎉 TSLA分析完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = analyze_tsla()
    
    if success:
        print(f"\n📝 如需更详细分析，请运行:")
        print(f"  python tsla_strategy_analysis.py")
        print(f"\n📊 如需Web界面，请运行:")
        print(f"  python run_dashboard.py")
    
    input(f"\n按Enter键退出...")

if __name__ == "__main__":
    main()
