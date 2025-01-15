#include "strategy/moving_average_strategy.hpp"
#include <numeric>
#include <spdlog/spdlog.h>

namespace trading {

MovingAverageStrategy::MovingAverageStrategy(size_t short_period, size_t long_period)
    : short_period_(short_period)
    , long_period_(long_period) {
    if (short_period >= long_period) {
        throw std::invalid_argument("Short period must be less than long period");
    }
}

void MovingAverageStrategy::onData(const MarketData& data) {
    prices_.push_back(data.price);
    if (prices_.size() > long_period_) {
        prices_.pop_front();
    }
}

std::optional<OrderRequest> MovingAverageStrategy::checkSignals() {
    if (prices_.size() < long_period_) {
        return std::nullopt;
    }
    
    auto short_ma = calculateMA({prices_.end() - short_period_, prices_.end()});
    auto long_ma = calculateMA({prices_.end() - long_period_, prices_.end()});
    
    if (short_ma > long_ma && !position_open_) {
        position_open_ = true;
        return OrderRequest{
            .side = OrderSide::BUY,
            .type = OrderType::MARKET,
            .quantity = 1.0  // This should come from position sizing
        };
    }
    else if (short_ma < long_ma && position_open_) {
        position_open_ = false;
        return OrderRequest{
            .side = OrderSide::SELL,
            .type = OrderType::MARKET,
            .quantity = 1.0
        };
    }
    
    return std::nullopt;
}

void MovingAverageStrategy::reset() {
    prices_.clear();
    position_open_ = false;
}

std::string MovingAverageStrategy::getName() const {
    return "Moving Average Strategy";
}

double MovingAverageStrategy::calculateMA(const std::deque<double>& prices) const {
    return std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
}

} // namespace trading 