#!/usr/bin/env python3
"""
强化学习策略优化器
Author: LDL
Date: 2025-01-25

使用强化学习自动优化交易策略参数和决策
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment(gym.Env):
    """交易环境 - 强化学习的训练环境"""
    
    def __init__(self, data, initial_balance=100000, transaction_cost=0.001):
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # 动作空间：0=持有, 1=买入, 2=卖出
        self.action_space = spaces.Discrete(3)
        
        # 状态空间：价格特征 + 持仓信息
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(10,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 20  # 从第20天开始，确保有足够的历史数据
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """获取当前状态观察"""
        if self.current_step >= len(self.data):
            return np.zeros(10)
        
        # 价格特征
        current_price = self.data.loc[self.current_step, 'Close']
        
        # 技术指标
        prices = self.data.loc[max(0, self.current_step-20):self.current_step, 'Close']
        
        if len(prices) < 2:
            return np.zeros(10)
        
        # 计算特征
        returns = prices.pct_change().fillna(0)
        
        features = [
            current_price / prices.iloc[0] - 1,  # 相对价格变化
            returns.iloc[-1],  # 最新收益率
            returns.mean(),  # 平均收益率
            returns.std(),  # 收益率标准差
            (current_price - prices.mean()) / prices.std(),  # 标准化价格位置
            len(prices[prices > current_price]) / len(prices),  # 价格分位数
            self.shares_held / 1000,  # 标准化持仓
            self.balance / self.initial_balance,  # 标准化余额
            self.net_worth / self.initial_balance,  # 标准化净值
            (self.net_worth - self.max_net_worth) / self.initial_balance  # 回撤
        ]
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action):
        """执行动作"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.loc[self.current_step, 'Close']
        
        # 执行动作
        reward = 0
        
        if action == 1:  # 买入
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))
        
        elif action == 2:  # 卖出
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trades.append(('SELL', self.current_step, current_price, self.shares_held))
                self.shares_held = 0
        
        # 更新净值
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # 移动到下一步
        self.current_step += 1
        
        # 计算奖励
        if self.current_step < len(self.data):
            next_price = self.data.loc[self.current_step, 'Close']
            price_change = (next_price - current_price) / current_price
            
            # 基于持仓和价格变化的奖励
            if self.shares_held > 0:
                reward = price_change * 100  # 持有时，价格上涨获得正奖励
            else:
                reward = -price_change * 50  # 空仓时，价格下跌获得正奖励
            
            # 添加风险调整
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            reward -= drawdown * 50  # 回撤惩罚
        
        # 检查是否结束
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held
        }

class DQNAgent:
    """深度Q网络智能体"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # 简化的Q网络（使用numpy实现）
        self.q_table = {}
        
    def remember(self, state, action, reward, next_state, done):
        """记住经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 简化的Q值计算
        state_key = tuple(np.round(state, 2))
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.random(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def replay(self, batch_size=32):
        """经验回放学习"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state_key = tuple(np.round(next_state, 2))
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = np.random.random(self.action_size)
                target = reward + 0.95 * np.amax(self.q_table[next_state_key])
            
            state_key = tuple(np.round(state, 2))
            if state_key not in self.q_table:
                self.q_table[state_key] = np.random.random(self.action_size)
            
            self.q_table[state_key][action] = target
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ReinforcementLearningOptimizer:
    """强化学习策略优化器"""
    
    def __init__(self):
        self.agent = None
        self.env = None
        self.training_history = []
        
        print("🤖 强化学习策略优化器初始化完成")
        print("Author: LDL")
    
    def create_training_environment(self, data, initial_balance=100000):
        """创建训练环境"""
        print("🏗️ 创建强化学习训练环境...")
        
        self.env = TradingEnvironment(data, initial_balance)
        self.agent = DQNAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n
        )
        
        print(f"✅ 环境创建完成")
        print(f"  状态空间维度: {self.env.observation_space.shape[0]}")
        print(f"  动作空间大小: {self.env.action_space.n}")
        print(f"  初始资金: ${initial_balance:,}")
    
    def train_agent(self, episodes=100):
        """训练强化学习智能体"""
        print(f"🎓 开始训练强化学习智能体 ({episodes}轮)...")
        
        if self.env is None or self.agent is None:
            print("❌ 请先创建训练环境")
            return
        
        scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # 经验回放学习
            if len(self.agent.memory) > 32:
                self.agent.replay(32)
            
            scores.append(total_reward)
            final_net_worth = info.get('net_worth', 0)
            
            if episode % 20 == 0:
                avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
                print(f"  轮次 {episode}: 平均奖励={avg_score:.2f}, 最终净值=${final_net_worth:,.2f}, 探索率={self.agent.epsilon:.3f}")
        
        self.training_history = scores
        print(f"✅ 训练完成！平均奖励: {np.mean(scores):.2f}")
        
        return scores
    
    def test_strategy(self, test_data):
        """测试训练好的策略"""
        print("🧪 测试强化学习策略...")
        
        if self.agent is None:
            print("❌ 请先训练智能体")
            return None
        
        # 创建测试环境
        test_env = TradingEnvironment(test_data)
        state = test_env.reset()
        
        actions_taken = []
        portfolio_values = []
        
        # 设置为测试模式（不探索）
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        while True:
            action = self.agent.act(state)
            next_state, reward, done, info = test_env.step(action)
            
            actions_taken.append(action)
            portfolio_values.append(info['net_worth'])
            
            state = next_state
            
            if done:
                break
        
        # 恢复探索率
        self.agent.epsilon = original_epsilon
        
        # 计算性能指标
        initial_value = test_env.initial_balance
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 计算基准收益（买入持有）
        initial_price = test_data.iloc[0]['Close']
        final_price = test_data.iloc[-1]['Close']
        benchmark_return = (final_price - initial_price) / initial_price
        
        results = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'actions_taken': actions_taken,
            'portfolio_values': portfolio_values,
            'trades': test_env.trades
        }
        
        print(f"📊 测试结果:")
        print(f"  初始资金: ${initial_value:,.2f}")
        print(f"  最终资金: ${final_value:,.2f}")
        print(f"  总收益率: {total_return:.2%}")
        print(f"  基准收益率: {benchmark_return:.2%}")
        print(f"  超额收益: {total_return - benchmark_return:.2%}")
        print(f"  交易次数: {len(test_env.trades)}")
        
        return results
    
    def generate_optimized_signals(self, current_data):
        """生成优化的交易信号"""
        print("🎯 生成强化学习优化信号...")
        
        if self.agent is None:
            print("❌ 请先训练智能体")
            return None
        
        # 创建临时环境获取当前状态
        temp_env = TradingEnvironment(current_data)
        state = temp_env._get_observation()
        
        # 获取动作建议
        self.agent.epsilon = 0  # 不探索，使用最佳策略
        action = self.agent.act(state)
        
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = action_map[action]
        
        # 计算置信度（基于Q值差异）
        state_key = tuple(np.round(state, 2))
        if state_key in self.agent.q_table:
            q_values = self.agent.q_table[state_key]
            max_q = np.max(q_values)
            second_max_q = np.partition(q_values, -2)[-2]
            confidence = min((max_q - second_max_q) * 100, 95)
        else:
            confidence = 50
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'q_values': q_values.tolist() if state_key in self.agent.q_table else [0, 0, 0],
            'state_features': state.tolist()
        }
        
        print(f"🚦 强化学习信号: {signal} (置信度: {confidence:.1f}%)")
        
        return result

def main():
    """演示强化学习优化功能"""
    print("🤖 强化学习策略优化演示")
    print("Author: LDL")
    print("="*50)
    
    # 生成演示数据
    print("\n📊 生成演示数据...")
    dates = pd.date_range(start='2024-01-01', end='2025-01-25', freq='D')
    dates = dates[dates.weekday < 5]
    
    np.random.seed(42)
    prices = [200]
    for i in range(1, len(dates)):
        change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 50))
    
    demo_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(20000000, 80000000, len(dates))
    })
    demo_data.set_index('Date', inplace=True)
    
    # 分割训练和测试数据
    split_point = int(len(demo_data) * 0.8)
    train_data = demo_data.iloc[:split_point]
    test_data = demo_data.iloc[split_point:]
    
    print(f"✅ 数据准备完成")
    print(f"  训练数据: {len(train_data)}天")
    print(f"  测试数据: {len(test_data)}天")
    
    # 创建优化器
    optimizer = ReinforcementLearningOptimizer()
    
    # 创建环境并训练
    optimizer.create_training_environment(train_data)
    scores = optimizer.train_agent(episodes=100)
    
    # 测试策略
    results = optimizer.test_strategy(test_data)
    
    # 生成当前信号
    current_signal = optimizer.generate_optimized_signals(demo_data.tail(50))
    
    print("\n🎉 强化学习优化演示完成!")
    print("\n💡 强化学习特点:")
    print("  ✅ 自动策略优化")
    print("  ✅ 环境适应学习")
    print("  ✅ 风险收益平衡")
    print("  ✅ 动态参数调整")

if __name__ == "__main__":
    main()
