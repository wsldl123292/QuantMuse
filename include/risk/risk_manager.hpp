#pragma once
#include "core/market_types.hpp"
#include <unordered_map>

namespace trading {

class RiskManager {
public:
    explicit RiskManager(const nlohmann::json& config);
    
    bool validateOrder(const OrderRequest& order);
    void updatePosition(const MarketData& data);
    void resetDaily();
    
private:
    struct PositionInfo {
        double current_size{0.0};
        double entry_price{0.0};
        double unrealized_pnl{0.0};
    };
    
    double max_position_size_;
    double stop_loss_percent_;
    double take_profit_percent_;
    double max_daily_loss_;
    double daily_pnl_{0.0};
    
    std::unordered_map<std::string, PositionInfo> positions_;
};

} // namespace trading 