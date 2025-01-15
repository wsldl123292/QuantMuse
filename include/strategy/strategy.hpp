#pragma once
#include "core/market_types.hpp"
#include <memory>
#include <string>

namespace trading {

class Strategy {
public:
    virtual ~Strategy() = default;
    
    virtual void onData(const MarketData& data) = 0;
    virtual std::optional<OrderRequest> checkSignals() = 0;
    virtual void reset() = 0;
    
    virtual std::string getName() const = 0;
};

} // namespace trading 