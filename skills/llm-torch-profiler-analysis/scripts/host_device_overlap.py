#!/usr/bin/env python3
"""
host_device_overlap.py
======================
判断推理 profiling 是 host bound 还是 device bound 的脚本。

原理
----
推理是 "CPU 下发 kernel -> 设备执行" 的流水线。判断瓶颈只需问：
    "设备空闲的时候，CPU 在忙吗？"
- 设备空闲 && CPU 忙  ->  host bound（CPU 喂不饱设备）
- 设备 忙            ->  device bound（设备执行太慢）

数据来源：Ascend profiler 输出的 ASCEND_PROFILER_OUTPUT/trace_view.json
这个文件同时包含两条时间线（按时间戳 ts 对齐，单位 us）：
  cat == "cpu_op"  且 name 是 aten::xxx / vllm::xxx  ->  CPU(host) 侧算子事件
  cat == ""        且 name 是 "Computing" / "Free"   ->  设备侧状态事件

用法
----
    python3 host_device_overlap.py \
        <ASCEND_PROFILER_OUTPUT/trace_view.json 路径>

输出
----
    设备 span / 设备忙占比
    host 算子总耗时
    设备空闲时刻开始执行的 host 算子（= 暴露出来的 host 瓶颈）及其 Top 列表
"""

import json
import sys
import bisect
from collections import defaultdict


def main(trace_path):
    # ---------- 1. 读入 trace_view.json，把 CPU 事件和设备事件分开 ----------
    with open(trace_path) as f:
        data = json.load(f)          # 整个文件是一个事件数组

    dev_busy = []   # 设备忙碌区间 [(start, end), ...]
    host = []       # host(cpu_op) 事件 [(start, dur, name), ...]

    for e in data:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue                  # 只关心 X 类型（complete event，有起止时间）
        try:
            ts = float(e["ts"])       # 事件开始时间（us）
            dur = float(e.get("dur", 0))
        except (KeyError, ValueError):
            continue
        name = e.get("name", "")
        cat = e.get("cat", "")

        if cat == "":
            # 设备侧事件：Computing = 设备在算；Free = 设备空闲
            if name == "Computing":
                dev_busy.append((ts, ts + dur))
        elif cat == "cpu_op" and dur > 0:
            host.append((ts, dur, name))

    # ---------- 2. 合并相邻的设备忙碌区间（同一个波里 kernel 连在一起） ----------
    dev_busy.sort()
    busy = []
    for s, e in dev_busy:
        if busy and s <= busy[-1][1]:          # 与上一个区间重叠或相接
            busy[-1][1] = max(busy[-1][1], e)  # 合并
        else:
            busy.append([s, e])

    # 设备整体统计
    span = busy[-1][1] - busy[0][0] if busy else 0
    busy_time = sum(b - a for a, b in busy)
    print("=" * 70)
    print(f"设备 span        : {span/1e6:7.2f} s")
    print(f"设备忙碌         : {busy_time/1e6:7.2f} s  = {busy_time/span*100:5.1f}%   (>=90% 才是 device bound)")
    print(f"设备空闲         : {(span-busy_time)/1e6:7.2f} s  = {(span-busy_time)/span*100:5.1f}%")
    print("=" * 70)

    # ---------- 3. 核心：对每个 host 算子，判断它开始时设备是否空闲 ----------
    # 用二分查找定位：这个时间点在哪个设备忙碌区间里？
    busy_starts = [b[0] for b in busy]

    def device_idle_at(t):
        i = bisect.bisect_right(busy_starts, t) - 1
        return i < 0 or t >= busy[i][1]        # 不在任何忙碌区间内 = 空闲

    total_host = 0.0        # host 算子总耗时
    exposed = 0.0           # 设备空闲时刻开始执行的 host 算子耗时
    exposed_by_op = defaultdict(float)         # 按算子名聚合
    n_exposed = 0

    for ts, dur, name in host:
        total_host += dur
        if device_idle_at(ts):                 # 设备正闲，CPU 却在这个时刻发起了算子
            exposed += dur
            exposed_by_op[name] += dur
            n_exposed += 1

    print(f"host(cpu_op) 算子总耗时 : {total_host/1e6:7.2f} s   (嵌套时间，仅看量级)")
    print(f"其中 {n_exposed} 个算子（共 {exposed/1e6:.2f}s）在设备空闲时刻开始")
    print(f"  => 设备空闲的 {(span-busy_time)/1e6:.2f}s 里，host 有 {exposed/1e6:.2f}s 在干活")
    print("  暴露出来的 host 瓶颈 Top 10（设备空闲时 CPU 在跑什么）:")
    for name, t in sorted(exposed_by_op.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name[:58]:60s} {t/1000:9.1f} ms")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
