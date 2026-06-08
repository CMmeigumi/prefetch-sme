/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 */
#pragma once

#include <cstdint>

namespace kutacc {

inline int get_thread_id()
{
    return 0;
}

inline int get_thread_num()
{
    return 1;
}

template <typename F>
inline void parallel_for(int64_t start, int64_t end, int64_t step, F &&func)
{
    for (int64_t i = start; i < end; i += step) {
        func(i, i + 1);
    }
}

} // namespace kutacc
