#pragma once
#include "core/market_types.hpp"
#include <functional>
#include <vector>
#include <memory>

namespace trading {

class DataFetcher {
public:
    using DataCallback = std::function<void(const MarketData&)>;
    
    virtual ~DataFetcher() = default;
    virtual void subscribeToRealtime(const std::string& symbol, DataCallback callback) = 0;
    virtual std::vector<MarketData> fetchHistorical(
        const std::string& symbol,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) = 0;
};

} // namespace trading 