#pragma once
#include <stdexcept>
#include <string>

namespace trading {

class TradingException : public std::runtime_error {
public:
    explicit TradingException(const std::string& message) 
        : std::runtime_error(message) {}
};

class DataFetchException : public TradingException {
public:
    explicit DataFetchException(const std::string& message) 
        : TradingException(message) {}
};

class OrderExecutionException : public TradingException {
public:
    explicit OrderExecutionException(const std::string& message) 
        : TradingException(message) {}
};

class ConfigurationException : public TradingException {
public:
    explicit ConfigurationException(const std::string& message) 
        : TradingException(message) {}
};

} // namespace trading 