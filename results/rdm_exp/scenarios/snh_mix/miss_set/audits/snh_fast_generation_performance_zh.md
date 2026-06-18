# snh fast generation performance

- 旧脚本 5% 约 3 分钟/天。
- 旧脚本 10% 约 10 分钟/天。
- 旧脚本 20% 超过 20 分钟/天。
- 本轮通过 anchor-neighbor 保护、chunk 外层循环、event_id 直接写入、分片 checkpoint 与原子写入提升速度与稳定性。

- 0.05 总耗时秒数: `229.571`
- 0.10 总耗时秒数: `423.952`
- 0.20 总耗时秒数: `920.184`
- 0.30 总耗时秒数: `1635.360`

- mean_seconds_per_chunk: `13.152`
- median_seconds_per_chunk: `10.910`
- max_seconds_per_chunk: `33.891`
