#pragma once
#include "data/data_fetcher.hpp"
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_client.hpp>

namespace trading {

class BinanceFetcher : public DataFetcher {
public:
    BinanceFetcher();
    ~BinanceFetcher() override;

    void subscribeToRealtime(const std::string& symbol, DataCallback callback) override;
    std::vector<MarketData> fetchHistorical(
        const std::string& symbol,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) override;

private:
    using WebsocketClient = websocketpp::client<websocketpp::config::asio_tls_client>;
    
    void initializeWebsocket();
    void handleMessage(const std::string& payload);

    WebsocketClient ws_client_;
    websocketpp::connection_hdl connection_;
    DataCallback callback_;
    bool is_connected_{false};
};

} // namespace trading 