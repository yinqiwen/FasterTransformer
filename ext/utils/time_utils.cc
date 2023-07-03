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
#include "time_utils.h"
#include <limits.h>
#include <sys/time.h>
#include <time.h>
// static const uint32 kmax_uint32 = 0xffffffff;

namespace ft_ext {
int64_t gettimeofday_us()
{
    struct timeval tv;
    uint64_t       ust;
    gettimeofday(&tv, nullptr);
    ust = ((int64_t)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust;
}
int64_t gettimeofday_ms()
{
    return gettimeofday_us() / 1000;
}
int64_t gettimeofday_s()
{
    return gettimeofday_us() / 1000000;
}
int64_t unixsec_from_timefield(const std::string& time_field)
{
    // const char *time_details = "16:35:12";
    struct tm tm;
    strptime(time_field.c_str(), "%Y-%m-%d %H:%M:%S", &tm);
    return fast_mktime(&tm);
}
time_t fast_mktime(struct tm* tm)
{
    static struct tm cache      = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    static time_t    time_cache = 0;
    // static time_t (*mktime_real)(struct tm *tm) = NULL;
    time_t result;
    time_t hmsarg;
    /* the epoch time portion of the request */
    hmsarg = 3600 * tm->tm_hour + 60 * tm->tm_min + tm->tm_sec;
    if (cache.tm_mday == tm->tm_mday && cache.tm_mon == tm->tm_mon && cache.tm_year == tm->tm_year) {
        /* cached - just add h,m,s from request to midnight */
        result = time_cache + hmsarg;
        /* Obscure, but documented, return value: only this value in arg struct.
         *
         * BUG: dst switchover was computed by mktime_real() for time 00:00:00
         * of arg day. So this return value WILL be wrong for switchover days
         * after the switchover occurs.  There is no clean way to detect this
         * situation in stock glibc.  This bug will be reflected in unit test
         * until fixed.  See also github issues #1 and #2.
         */
        tm->tm_isdst = cache.tm_isdst;
    }
    else {
        /* not cached - recompute midnight on requested day */
        cache.tm_mday = tm->tm_mday;
        cache.tm_mon  = tm->tm_mon;
        cache.tm_year = tm->tm_year;
        time_cache    = mktime(&cache);
        tm->tm_isdst  = cache.tm_isdst;
        result        = (-1 == time_cache) ? -1 : time_cache + hmsarg;
    }
    return result;
}
}  // namespace ft_ext