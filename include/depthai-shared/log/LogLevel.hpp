#pragma once

// std
#include <cstdint>

// libraries
#include <nlohmann/json.hpp>

#if SPDLOG_VERSION < 10601
#include "spdlog/spdlog.h"
#endif

namespace dai {

// Work around when using v1.5.0 of SPDLOG library
#if SPDLOG_VERSION < 10601
extern spdlog::level::level_enum CurrentLogLevel;
#endif

// Follows spdlog levels
enum class LogLevel : std::int32_t { TRACE = 0, DEBUG, INFO, WARN, ERR, CRITICAL, OFF };

}  // namespace dai
