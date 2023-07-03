/*
 * Copyright (c) 2022 TENCENT CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "spdlog/spdlog.h"
#include <memory>

#include "fmt/ostream.h"

namespace ft_ext {
extern std::shared_ptr<spdlog::logger> g_default_looger;
void                                   set_default_logger(std::shared_ptr<spdlog::logger> logger);
spdlog::logger*                        get_default_raw_logger();
}  // namespace ft_ext

#define FT_DEBUG(...)                                                                                                  \
    do {                                                                                                               \
        auto _local_logger_ =                                                                                          \
            ft_ext::g_default_looger ? ft_ext::g_default_looger.get() : spdlog::default_logger_raw();                  \
        if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::debug)) {                           \
            SPDLOG_LOGGER_DEBUG(_local_logger_, __VA_ARGS__);                                                          \
        }                                                                                                              \
    } while (0)

#define FT_INFO(...)                                                                                                   \
    do {                                                                                                               \
        auto _local_logger_ =                                                                                          \
            ft_ext::g_default_looger ? ft_ext::g_default_looger.get() : spdlog::default_logger_raw();                  \
        if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::info)) {                            \
            SPDLOG_LOGGER_INFO(_local_logger_, __VA_ARGS__);                                                           \
        }                                                                                                              \
    } while (0)

#define FT_WARN(...)                                                                                                   \
    do {                                                                                                               \
        auto _local_logger_ =                                                                                          \
            ft_ext::g_default_looger ? ft_ext::g_default_looger.get() : spdlog::default_logger_raw();                  \
        if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::warn)) {                            \
            SPDLOG_LOGGER_WARN(_local_logger_, __VA_ARGS__);                                                           \
        }                                                                                                              \
    } while (0)

#define FT_ERROR(...)                                                                                                  \
    do {                                                                                                               \
        auto _local_logger_ =                                                                                          \
            ft_ext::g_default_looger ? ft_ext::g_default_looger.get() : spdlog::default_logger_raw();                  \
        if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::err)) {                             \
            SPDLOG_LOGGER_ERROR(_local_logger_, __VA_ARGS__);                                                          \
        }                                                                                                              \
    } while (0)

#define FT_CRITICAL(...)                                                                                               \
    do {                                                                                                               \
        auto _local_logger_ =                                                                                          \
            ft_ext::g_default_looger ? ft_ext::g_default_looger.get() : spdlog::default_logger_raw();                  \
        if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::critical)) {                        \
            SPDLOG_LOGGER_CRITICAL(_local_logger_, __VA_ARGS__);                                                       \
        }                                                                                                              \
    } while (0)
