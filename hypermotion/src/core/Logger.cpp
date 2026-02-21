#include "HyperMotion/core/Logger.h"

#include <iomanip>

namespace hm {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::Logger() = default;

void Logger::setLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

LogLevel Logger::getLevel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return level_;
}

void Logger::setCallback(LogCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(callback);
}

void Logger::log(LogLevel level, const std::string& tag, const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level < level_) return;
    }

    LogCallback cb;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cb = callback_;
    }

    if (cb) {
        cb(level, tag, message);
    } else {
        std::ostringstream oss;
        oss << "[" << timestamp() << "] "
            << "[" << levelToString(level) << "] "
            << "[" << tag << "] "
            << message;

        std::lock_guard<std::mutex> lock(mutex_);
        if (level >= LogLevel::Error) {
            std::cerr << oss.str() << std::endl;
        } else {
            std::cout << oss.str() << std::endl;
        }
    }
}

void Logger::trace(const std::string& tag, const std::string& message) { log(LogLevel::Trace, tag, message); }
void Logger::debug(const std::string& tag, const std::string& message) { log(LogLevel::Debug, tag, message); }
void Logger::info(const std::string& tag, const std::string& message)  { log(LogLevel::Info, tag, message); }
void Logger::warn(const std::string& tag, const std::string& message)  { log(LogLevel::Warn, tag, message); }
void Logger::error(const std::string& tag, const std::string& message) { log(LogLevel::Error, tag, message); }
void Logger::fatal(const std::string& tag, const std::string& message) { log(LogLevel::Fatal, tag, message); }

void Logger::setProgressCallback(ProgressCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    progressCallback_ = std::move(cb);
}

void Logger::reportProgress(float percent, const std::string& message) {
    ProgressCallback cb;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cb = progressCallback_;
    }

    if (cb) {
        cb(percent, message);
    } else {
        std::ostringstream oss;
        oss << "[PROGRESS] " << std::fixed << std::setprecision(1) << percent << "% - " << message;
        info("Progress", oss.str());
    }
}

const char* Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Fatal: return "FATAL";
        case LogLevel::Off:   return "OFF  ";
    }
    return "?????";
}

std::string Logger::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_buf{};
    localtime_r(&time, &tm_buf);

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

} // namespace hm
