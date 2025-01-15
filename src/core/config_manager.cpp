#include "core/config_manager.hpp"
#include "core/exceptions.hpp"
#include <fstream>
#include <spdlog/spdlog.h>

namespace trading {

ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

void ConfigManager::loadConfig(const std::string& config_path) {
    try {
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw ConfigurationException("Unable to open config file: " + config_path);
        }
        config_file >> config_;
    } catch (const nlohmann::json::exception& e) {
        throw ConfigurationException("Error parsing config file: " + std::string(e.what()));
    }
    spdlog::info("Configuration loaded successfully from {}", config_path);
}

nlohmann::json ConfigManager::getConfig(const std::string& key) const {
    try {
        return config_.at(key);
    } catch (const nlohmann::json::exception& e) {
        throw ConfigurationException("Error accessing config key '" + key + "': " + e.what());
    }
}

} // namespace trading 