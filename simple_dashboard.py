#!/usr/bin/env python3
"""
简化的TSLA分析仪表板
Author: LDL
Date: 2025-01-25

使用Streamlit创建简单的TSLA分析界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="TSLA量化分析仪表板",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_mock_tsla_data():
    """生成模拟的TSLA数据"""
    # 生成过去一年的日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 过滤掉周末
    dates = dates[dates.weekday < 5]
    
    # 生成模拟价格数据
    np.random.seed(42)
    initial_price = 200.0
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 50))
    
    # 生成OHLCV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_volatility = 0.02
        high = close * (1 + np.random.uniform(0, daily_volatility))
        low = close * (1 - np.random.uniform(0, daily_volatility))
        
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.uniform(-0.01, 0.01))
        
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

def calculate_technical_indicators(df):
    """计算技术指标"""
    # 移动平均线
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 布林带
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def create_price_chart(df):
    """创建价格图表"""
    fig = go.Figure()
    
    # K线图
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='TSLA'
    ))
    
    # 移动平均线
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA20'],
        mode='lines',
        name='MA20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA50'],
        mode='lines',
        name='MA50',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title='TSLA 价格走势图',
        yaxis_title='价格 ($)',
        xaxis_title='日期',
        height=500
    )
    
    return fig

def create_volume_chart(df):
    """创建成交量图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='成交量',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='TSLA 成交量',
        yaxis_title='成交量',
        xaxis_title='日期',
        height=300
    )
    
    return fig

def create_rsi_chart(df):
    """创建RSI图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # 添加超买超卖线
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
    
    fig.update_layout(
        title='TSLA RSI指标',
        yaxis_title='RSI',
        xaxis_title='日期',
        height=300,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def main():
    """主函数"""
    # 标题
    st.title("🚗 TSLA量化分析仪表板")
    st.markdown("**Author: LDL** | 使用模拟数据进行演示")
    
    # 侧边栏
    st.sidebar.header("📊 分析设置")
    
    # 数据时间范围选择
    time_range = st.sidebar.selectbox(
        "选择时间范围",
        ["最近30天", "最近90天", "最近180天", "最近一年"],
        index=3
    )
    
    # 技术指标选择
    show_ma = st.sidebar.checkbox("显示移动平均线", value=True)
    show_volume = st.sidebar.checkbox("显示成交量", value=True)
    show_rsi = st.sidebar.checkbox("显示RSI指标", value=True)
    
    # 生成数据
    with st.spinner("正在生成数据..."):
        df = generate_mock_tsla_data()
        df = calculate_technical_indicators(df)
    
    # 根据时间范围过滤数据
    days_map = {"最近30天": 30, "最近90天": 90, "最近180天": 180, "最近一年": 365}
    days = days_map[time_range]
    df_filtered = df.tail(days)
    
    # 主要指标
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df_filtered['Close'].iloc[-1]
    price_change = (current_price - df_filtered['Close'].iloc[0]) / df_filtered['Close'].iloc[0]
    volume = df_filtered['Volume'].iloc[-1]
    rsi = df_filtered['RSI'].iloc[-1]
    
    with col1:
        st.metric("当前价格", f"${current_price:.2f}", f"{price_change:.2%}")
    
    with col2:
        st.metric("最高价", f"${df_filtered['High'].max():.2f}")
    
    with col3:
        st.metric("最低价", f"${df_filtered['Low'].min():.2f}")
    
    with col4:
        st.metric("RSI", f"{rsi:.1f}", "正常" if 30 < rsi < 70 else ("超买" if rsi > 70 else "超卖"))
    
    # 图表区域
    st.subheader("📈 价格分析")
    price_chart = create_price_chart(df_filtered)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # 两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        if show_volume:
            st.subheader("📊 成交量分析")
            volume_chart = create_volume_chart(df_filtered)
            st.plotly_chart(volume_chart, use_container_width=True)
    
    with col2:
        if show_rsi:
            st.subheader("🧮 RSI技术指标")
            rsi_chart = create_rsi_chart(df_filtered)
            st.plotly_chart(rsi_chart, use_container_width=True)
    
    # 数据表格
    st.subheader("📋 最新数据")
    st.dataframe(df_filtered.tail(10).round(2), use_container_width=True)
    
    # 分析总结
    st.subheader("💡 分析总结")
    
    # 生成简单的分析
    ma20 = df_filtered['MA20'].iloc[-1]
    ma50 = df_filtered['MA50'].iloc[-1]
    
    analysis = []
    if current_price > ma20:
        analysis.append("✅ 价格在20日均线上方，短期趋势向上")
    else:
        analysis.append("❌ 价格在20日均线下方，短期趋势向下")
    
    if ma20 > ma50:
        analysis.append("✅ 20日均线在50日均线上方，中期趋势向上")
    else:
        analysis.append("❌ 20日均线在50日均线下方，中期趋势向下")
    
    if 30 < rsi < 70:
        analysis.append("✅ RSI处于正常区间，无超买超卖")
    elif rsi > 70:
        analysis.append("⚠️ RSI超买，可能面临回调压力")
    else:
        analysis.append("⚠️ RSI超卖，可能存在反弹机会")
    
    for item in analysis:
        st.write(item)
    
    # 免责声明
    st.markdown("---")
    st.warning("⚠️ 本仪表板使用模拟数据，仅供演示和学习使用。投资有风险，决策需谨慎。")

if __name__ == "__main__":
    main()
