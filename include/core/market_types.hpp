#pragma once
#include <string>
#include <chrono>
#include <vector>
#include <optional>

namespace trading {

enum class MarketType {
    CRYPTO,
    STOCK
};

enum class OrderSide {
    BUY,
    SELL
};

enum class OrderType {
    MARKET,
    LIMIT
};

enum class OrderStatus {
    PENDING,
    FILLED,
    CANCELLED,
    REJECTED
};

struct MarketData {
    std::string symbol;
    MarketType market_type;
    double price;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    
    // OHLC data
    double open;
    double high;
    double low;
    double close;
};

struct OrderRequest {
    std::string symbol;
    MarketType market_type;
    OrderSide side;
    OrderType type;
    double quantity;
    std::optional<double> price;  // Optional for market orders
};

struct OrderResponse {
    std::string order_id;
    OrderStatus status;
    std::string error_message;
    std::chrono::system_clock::time_point timestamp;
};

} // namespace trading 