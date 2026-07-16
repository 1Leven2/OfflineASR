#pragma once

#include <string>

namespace offline_asr {

/**
 * 日志工具 — 基于 spdlog 的宏封装
 *
 * 使用:
 *   LOG_INFO("Model loaded: {}", model_path);
 *   LOG_WARN("Beam size too small: {}", beam_size);
 *   LOG_ERROR("Failed to open: {}", audio_path);
 */

void InitLogger(const std::string& level = "info");

}  // namespace offline_asr

// 日志宏
#include <spdlog/spdlog.h>

#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#define LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
#define LOG_INFO(...)  spdlog::info(__VA_ARGS__)
#define LOG_WARN(...)  spdlog::warn(__VA_ARGS__)
#define LOG_ERROR(...) spdlog::error(__VA_ARGS__)
