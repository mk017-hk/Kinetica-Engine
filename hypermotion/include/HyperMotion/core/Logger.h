#pragma once

#include <string>
#include <functional>
#include <chrono>
#include <sstream>
#include <iostream>
#include <mutex>

namespace hm {

enum class LogLevel : int {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
    Off = 6
};

using LogCallback = std::function<void(LogLevel, const std::string&, const std::string&)>;

class Logger {
public:
    static Logger& instance();

    void setLevel(LogLevel level);
    LogLevel getLevel() const;

    void setCallback(LogCallback callback);

    void log(LogLevel level, const std::string& tag, const std::string& message);

    void trace(const std::string& tag, const std::string& message);
    void debug(const std::string& tag, const std::string& message);
    void info(const std::string& tag, const std::string& message);
    void warn(const std::string& tag, const std::string& message);
    void error(const std::string& tag, const std::string& message);
    void fatal(const std::string& tag, const std::string& message);

    // Progress reporting
    using ProgressCallback = std::function<void(float percent, const std::string& message)>;
    void setProgressCallback(ProgressCallback cb);
    void reportProgress(float percent, const std::string& message);

private:
    Logger();
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    LogLevel level_ = LogLevel::Info;
    LogCallback callback_;
    ProgressCallback progressCallback_;
    mutable std::mutex mutex_;

    static const char* levelToString(LogLevel level);
    static std::string timestamp();
};

// Convenience macros
#define HM_LOG_TRACE(tag, msg) ::hm::Logger::instance().trace(tag, msg)
#define HM_LOG_DEBUG(tag, msg) ::hm::Logger::instance().debug(tag, msg)
#define HM_LOG_INFO(tag, msg)  ::hm::Logger::instance().info(tag, msg)
#define HM_LOG_WARN(tag, msg)  ::hm::Logger::instance().warn(tag, msg)
#define HM_LOG_ERROR(tag, msg) ::hm::Logger::instance().error(tag, msg)
#define HM_LOG_FATAL(tag, msg) ::hm::Logger::instance().fatal(tag, msg)

} // namespace hm
