#pragma once
#include "strategy/strategy.hpp"
#include <deque>

namespace trading {

class MovingAverageStrategy : public Strategy {
public:
    MovingAverageStrategy(size_t short_period = 10, size_t long_period = 30);
    
    void onData(const MarketData& data) override;
    std::optional<OrderRequest> checkSignals() override;
    void reset() override;
    std::string getName() const override;

private:
    double calculateMA(const std::deque<double>& prices) const;
    
    size_t short_period_;
    size_t long_period_;
    std::deque<double> prices_;
    bool position_open_{false};
};

} // namespace trading 