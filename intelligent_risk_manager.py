#!/usr/bin/env python3
"""
智能风险管理系统
Author: LDL
Date: 2025-01-25

使用AI技术进行智能风险评估、动态仓位管理和自动止损
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class IntelligentRiskManager:
    """智能风险管理器"""
    
    def __init__(self, initial_capital=100000, max_position_size=0.1, max_portfolio_risk=0.02):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size  # 单个仓位最大占比
        self.max_portfolio_risk = max_portfolio_risk  # 投资组合最大风险
        
        self.risk_models = {}
        self.anomaly_detector = None
        self.risk_history = []
        
        print("🛡️ 智能风险管理系统初始化完成")
        print("Author: LDL")
        print(f"  初始资金: ${initial_capital:,}")
        print(f"  最大单仓位: {max_position_size:.1%}")
        print(f"  最大组合风险: {max_portfolio_risk:.1%}")
    
    def calculate_var(self, returns, confidence_level=0.05):
        """计算风险价值(VaR)"""
        if len(returns) < 30:
            return 0
        
        # 历史模拟法
        historical_var = np.percentile(returns, confidence_level * 100)
        
        # 参数法（假设正态分布）
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        parametric_var = stats.norm.ppf(confidence_level, mean_return, std_return)
        
        # 取两者的平均值
        var = (historical_var + parametric_var) / 2
        
        return var
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """计算条件风险价值(CVaR)"""
        if len(returns) < 30:
            return 0
        
        var = self.calculate_var(returns, confidence_level)
        # CVaR是超过VaR的损失的期望值
        tail_losses = returns[returns <= var]
        
        if len(tail_losses) > 0:
            cvar = np.mean(tail_losses)
        else:
            cvar = var
        
        return cvar
    
    def calculate_maximum_drawdown(self, portfolio_values):
        """计算最大回撤"""
        if len(portfolio_values) < 2:
            return 0
        
        cumulative_returns = np.array(portfolio_values)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """计算夏普比率"""
        if len(returns) < 2:
            return 0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252  # 日化无风险利率
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0
        
        sharpe_ratio = excess_returns / volatility * np.sqrt(252)  # 年化
        return sharpe_ratio
    
    def detect_market_anomalies(self, market_data):
        """检测市场异常"""
        print("🔍 检测市场异常...")
        
        # 准备特征
        features = []
        
        # 价格特征
        returns = market_data['Close'].pct_change().fillna(0)
        features.append(returns.values)
        
        # 波动率特征
        volatility = returns.rolling(20).std().fillna(0)
        features.append(volatility.values)
        
        # 成交量特征
        volume_change = market_data['Volume'].pct_change().fillna(0)
        features.append(volume_change.values)
        
        # 价格跳跃
        price_jumps = np.abs(returns) > 2 * volatility
        features.append(price_jumps.astype(float).values)
        
        # 组合特征矩阵
        X = np.column_stack(features)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练异常检测模型
        if self.anomaly_detector is None:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # 假设10%的数据是异常
                random_state=42
            )
            self.anomaly_detector.fit(X_scaled)
        
        # 检测异常
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        anomalies = self.anomaly_detector.predict(X_scaled)
        
        # 识别最近的异常
        recent_anomalies = []
        for i in range(max(0, len(anomalies) - 10), len(anomalies)):
            if anomalies[i] == -1:  # -1表示异常
                recent_anomalies.append({
                    'date': market_data.index[i],
                    'anomaly_score': anomaly_scores[i],
                    'return': returns.iloc[i],
                    'volatility': volatility.iloc[i]
                })
        
        if recent_anomalies:
            print(f"⚠️ 检测到 {len(recent_anomalies)} 个近期市场异常")
            for anomaly in recent_anomalies[-3:]:  # 显示最近3个
                print(f"  {anomaly['date'].date()}: 异常分数={anomaly['anomaly_score']:.3f}")
        else:
            print("✅ 未检测到显著市场异常")
        
        return recent_anomalies
    
    def calculate_optimal_position_size(self, expected_return, volatility, current_portfolio_value):
        """计算最优仓位大小"""
        # Kelly公式的修改版本
        if volatility <= 0:
            return 0
        
        # Kelly比例
        kelly_fraction = expected_return / (volatility ** 2)
        
        # 限制Kelly比例以控制风险
        kelly_fraction = min(kelly_fraction, 0.25)  # 最大25%
        kelly_fraction = max(kelly_fraction, 0)     # 不允许负值
        
        # 考虑最大仓位限制
        max_position_value = current_portfolio_value * self.max_position_size
        kelly_position_value = current_portfolio_value * kelly_fraction
        
        optimal_position_value = min(max_position_value, kelly_position_value)
        
        return optimal_position_value / current_portfolio_value
    
    def assess_portfolio_risk(self, positions, market_data):
        """评估投资组合风险"""
        print("📊 评估投资组合风险...")
        
        if not positions:
            return {
                'total_risk': 0,
                'var_1d': 0,
                'cvar_1d': 0,
                'max_drawdown': 0,
                'risk_level': 'LOW'
            }
        
        # 计算投资组合收益率
        portfolio_returns = []
        portfolio_values = []
        
        for i in range(1, len(market_data)):
            daily_return = 0
            portfolio_value = 0
            
            for symbol, position in positions.items():
                if symbol in market_data.columns:
                    price_change = (market_data[symbol].iloc[i] - market_data[symbol].iloc[i-1]) / market_data[symbol].iloc[i-1]
                    daily_return += position['weight'] * price_change
                    portfolio_value += position['value']
            
            portfolio_returns.append(daily_return)
            portfolio_values.append(portfolio_value)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # 计算风险指标
        var_1d = self.calculate_var(portfolio_returns, 0.05)
        cvar_1d = self.calculate_cvar(portfolio_returns, 0.05)
        max_drawdown = self.calculate_maximum_drawdown(portfolio_values)
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
        
        # 总体风险评分
        risk_score = 0
        risk_score += abs(var_1d) * 100  # VaR贡献
        risk_score += max_drawdown * 50   # 回撤贡献
        risk_score += max(0, 2 - sharpe_ratio) * 20  # 夏普比率贡献
        
        # 风险等级
        if risk_score < 5:
            risk_level = 'LOW'
        elif risk_score < 15:
            risk_level = 'MEDIUM'
        elif risk_score < 30:
            risk_level = 'HIGH'
        else:
            risk_level = 'EXTREME'
        
        risk_assessment = {
            'total_risk': risk_score,
            'var_1d': var_1d,
            'cvar_1d': cvar_1d,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'risk_level': risk_level,
            'portfolio_volatility': np.std(portfolio_returns) * np.sqrt(252)
        }
        
        print(f"📈 风险评估结果:")
        print(f"  风险等级: {risk_level}")
        print(f"  风险评分: {risk_score:.2f}")
        print(f"  1日VaR: {var_1d:.2%}")
        print(f"  1日CVaR: {cvar_1d:.2%}")
        print(f"  最大回撤: {max_drawdown:.2%}")
        print(f"  夏普比率: {sharpe_ratio:.2f}")
        
        return risk_assessment
    
    def generate_risk_alerts(self, risk_assessment, market_anomalies):
        """生成风险预警"""
        alerts = []
        
        # VaR预警
        if abs(risk_assessment['var_1d']) > self.max_portfolio_risk:
            alerts.append({
                'type': 'VAR_BREACH',
                'severity': 'HIGH',
                'message': f"VaR超过限制: {risk_assessment['var_1d']:.2%} > {self.max_portfolio_risk:.2%}",
                'recommendation': '建议减少仓位或增加对冲'
            })
        
        # 回撤预警
        if risk_assessment['max_drawdown'] > 0.1:  # 10%回撤
            alerts.append({
                'type': 'DRAWDOWN_WARNING',
                'severity': 'MEDIUM',
                'message': f"最大回撤过大: {risk_assessment['max_drawdown']:.2%}",
                'recommendation': '考虑止损或调整策略'
            })
        
        # 夏普比率预警
        if risk_assessment['sharpe_ratio'] < 0.5:
            alerts.append({
                'type': 'POOR_RISK_RETURN',
                'severity': 'MEDIUM',
                'message': f"风险调整收益较低: 夏普比率 {risk_assessment['sharpe_ratio']:.2f}",
                'recommendation': '重新评估投资策略'
            })
        
        # 市场异常预警
        if market_anomalies:
            alerts.append({
                'type': 'MARKET_ANOMALY',
                'severity': 'HIGH',
                'message': f"检测到 {len(market_anomalies)} 个市场异常",
                'recommendation': '密切监控市场，考虑降低风险敞口'
            })
        
        return alerts
    
    def suggest_risk_actions(self, alerts, current_positions):
        """建议风险管理行动"""
        actions = []
        
        for alert in alerts:
            if alert['type'] == 'VAR_BREACH':
                actions.append({
                    'action': 'REDUCE_POSITION',
                    'priority': 'HIGH',
                    'description': '减少高风险仓位至安全水平',
                    'target_reduction': '20-30%'
                })
            
            elif alert['type'] == 'DRAWDOWN_WARNING':
                actions.append({
                    'action': 'SET_STOP_LOSS',
                    'priority': 'MEDIUM',
                    'description': '为所有仓位设置止损点',
                    'stop_loss_level': '5-8%'
                })
            
            elif alert['type'] == 'MARKET_ANOMALY':
                actions.append({
                    'action': 'INCREASE_MONITORING',
                    'priority': 'HIGH',
                    'description': '增加市场监控频率，准备应急措施',
                    'monitoring_frequency': '每小时'
                })
        
        return actions

def main():
    """演示智能风险管理功能"""
    print("🛡️ 智能风险管理系统演示")
    print("Author: LDL")
    print("="*50)
    
    # 创建风险管理器
    risk_manager = IntelligentRiskManager(
        initial_capital=100000,
        max_position_size=0.15,
        max_portfolio_risk=0.02
    )
    
    # 生成演示数据
    print("\n📊 生成演示数据...")
    dates = pd.date_range(start='2024-01-01', end='2025-01-25', freq='D')
    dates = dates[dates.weekday < 5]
    
    np.random.seed(42)
    
    # 模拟TSLA价格（包含一些异常波动）
    prices = [200]
    for i in range(1, len(dates)):
        # 正常波动
        change = np.random.normal(0.001, 0.02)
        
        # 偶尔添加异常波动
        if np.random.random() < 0.05:  # 5%概率
            change += np.random.choice([-0.1, 0.1])  # ±10%的异常波动
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 50))
    
    demo_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(20000000, 80000000, len(dates))
    })
    demo_data.set_index('Date', inplace=True)
    
    print(f"✅ 生成了{len(demo_data)}条演示数据")
    
    # 检测市场异常
    anomalies = risk_manager.detect_market_anomalies(demo_data)
    
    # 模拟投资组合
    mock_positions = {
        'TSLA': {
            'weight': 0.6,
            'value': 60000
        },
        'AAPL': {
            'weight': 0.4,
            'value': 40000
        }
    }
    
    # 评估投资组合风险
    risk_assessment = risk_manager.assess_portfolio_risk(mock_positions, demo_data)
    
    # 生成风险预警
    alerts = risk_manager.generate_risk_alerts(risk_assessment, anomalies)
    
    if alerts:
        print(f"\n⚠️ 风险预警 ({len(alerts)}个):")
        for alert in alerts:
            print(f"  {alert['severity']}: {alert['message']}")
            print(f"    建议: {alert['recommendation']}")
    else:
        print("\n✅ 当前风险水平可控，无需特别预警")
    
    # 建议风险管理行动
    if alerts:
        actions = risk_manager.suggest_risk_actions(alerts, mock_positions)
        print(f"\n🎯 建议行动 ({len(actions)}个):")
        for action in actions:
            print(f"  {action['priority']}: {action['description']}")
    
    print("\n🎉 智能风险管理演示完成!")
    print("\n💡 智能风险管理特点:")
    print("  ✅ 实时风险监控")
    print("  ✅ 异常检测预警")
    print("  ✅ 动态仓位优化")
    print("  ✅ 智能止损建议")
    print("  ✅ 多维度风险评估")

if __name__ == "__main__":
    main()
