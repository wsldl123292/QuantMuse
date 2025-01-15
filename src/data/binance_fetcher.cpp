#include "data/binance_fetcher.hpp"
#include "core/exceptions.hpp"
#include "core/config_manager.hpp"
#include <curl/curl.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace trading {

BinanceFetcher::BinanceFetcher() {
    initializeWebsocket();
}

BinanceFetcher::~BinanceFetcher() {
    if (is_connected_) {
        ws_client_.stop();
    }
}

void BinanceFetcher::initializeWebsocket() {
    ws_client_.set_access_channels(websocketpp::log::alevel::none);
    ws_client_.set_error_channels(websocketpp::log::elevel::fatal);
    
    ws_client_.init_asio();
    
    ws_client_.set_message_handler([this](auto, auto msg) {
        handleMessage(msg->get_payload());
    });
}

void BinanceFetcher::subscribeToRealtime(const std::string& symbol, DataCallback callback) {
    callback_ = std::move(callback);
    
    auto& config = ConfigManager::getInstance();
    std::string ws_url = config.getConfig("data_sources")["crypto"]["binance"]["ws_url"];
    
    try {
        websocketpp::lib::error_code ec;
        auto conn = ws_client_.get_connection(ws_url, ec);
        
        if (ec) {
            throw DataFetchException("Could not create connection: " + ec.message());
        }
        
        connection_ = conn->get_handle();
        ws_client_.connect(conn);
        
        // Start the ASIO io_service run loop
        ws_client_.run();
        is_connected_ = true;
        
    } catch (const websocketpp::exception& e) {
        throw DataFetchException("Websocket error: " + std::string(e.what()));
    }
}

std::vector<MarketData> BinanceFetcher::fetchHistorical(
    const std::string& symbol,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {
    
    auto& config = ConfigManager::getInstance();
    std::string api_url = config.getConfig("data_sources")["crypto"]["binance"]["api_url"];
    
    // Implementation for REST API historical data fetch
    // ... (HTTP request implementation using CURL)
    
    return std::vector<MarketData>{};
}

void BinanceFetcher::handleMessage(const std::string& payload) {
    try {
        auto json = nlohmann::json::parse(payload);
        
        MarketData data{
            .symbol = json["s"],
            .market_type = MarketType::CRYPTO,
            .price = std::stod(json["p"].get<std::string>()),
            .volume = std::stod(json["q"].get<std::string>()),
            .timestamp = std::chrono::system_clock::now()
        };
        
        if (callback_) {
            callback_(data);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("Error parsing websocket message: {}", e.what());
    }
}

} // namespace trading 