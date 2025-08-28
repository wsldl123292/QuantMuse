#!/usr/bin/env python3
"""
TSLA 演示分析脚本
Author: LDL
Date: 2025-01-25

使用模拟数据演示TSLA分析功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_mock_tsla_data():
    """生成模拟的TSLA数据"""
    # 生成过去一年的日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 过滤掉周末
    dates = dates[dates.weekday < 5]
    
    # 生成模拟价格数据 (基于TSLA的大致价格范围)
    np.random.seed(42)  # 确保结果可重复
    
    # 初始价格
    initial_price = 200.0
    
    # 生成随机价格变化
    returns = np.random.normal(0.001, 0.03, len(dates))  # 日收益率
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 50))  # 价格不低于50
    
    # 生成OHLCV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # 生成开高低收
        daily_volatility = 0.02
        high = close * (1 + np.random.uniform(0, daily_volatility))
        low = close * (1 - np.random.uniform(0, daily_volatility))
        
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.uniform(-0.01, 0.01))
        
        # 生成成交量
        volume = np.random.randint(20000000, 80000000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def analyze_tsla_demo():
    """TSLA演示分析"""
    print("🚗 TSLA演示分析开始...")
    print("Author: LDL")
    print("⚠️  注意：使用模拟数据进行演示")
    print("="*50)
    
    try:
        # 1. 生成模拟数据
        print("📊 正在生成模拟TSLA数据...")
        
        hist_data = generate_mock_tsla_data()
        
        print(f"✅ 成功生成 {len(hist_data)} 条数据记录")
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
        
        # 8. 模拟公司信息
        print(f"\n🏢 公司信息 (模拟):")
        print(f"  公司名称: Tesla, Inc.")
        print(f"  行业: Auto Manufacturers")
        print(f"  市值: $800,000,000,000")
        print(f"  P/E比率: 65.4")
        print(f"  Beta系数: 2.1")
        
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
        
        # 10. 演示说明
        print(f"\n🎯 演示功能说明:")
        print(f"  ✅ 数据获取与处理")
        print(f"  ✅ 技术指标计算 (MA, RSI)")
        print(f"  ✅ 风险指标分析")
        print(f"  ✅ 交易信号生成")
        print(f"  ✅ 成交量分析")
        print(f"  ✅ 综合评价系统")
        
        # 11. 风险提示
        print(f"\n⚠️  重要提示:")
        print(f"  • 本演示使用模拟数据，仅展示分析功能")
        print(f"  • 实际使用需要真实市场数据")
        print(f"  • 投资有风险，决策需谨慎")
        
        print(f"\n🎉 TSLA演示分析完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = analyze_tsla_demo()
    
    if success:
        print(f"\n📝 系统功能演示:")
        print(f"  ✅ 数据获取和处理模块")
        print(f"  ✅ 技术分析计算引擎")
        print(f"  ✅ 量化指标评估系统")
        print(f"  ✅ 交易信号生成器")
        print(f"  ✅ 风险评估框架")
        
        print(f"\n🔧 下一步:")
        print(f"  • 配置真实数据源API")
        print(f"  • 运行完整策略分析")
        print(f"  • 启动Web可视化界面")
    
    input(f"\n按Enter键退出...")

if __name__ == "__main__":
    main()
