#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
⚠️⚠️ 本脚本已弃用 —— 请勿运行 ⚠️⚠️

日期：2026-09-01

## 为什么弃用

本脚本最初的假设是：`data/Defect_dataset` 中未被 `Defect_dataset_16_rebuilt`
采用的 399 张图像是"带标签、从未使用的检测车原图"，可以用零成本扩充评测基准。

**该假设经进一步核查被证伪（2026-09-01 当日修正）：**

| 项 | 事实 |
|---|---|
| 那 399 张是什么 | **全部是离线增强副本**：`_aug0_blur` 66 / `_aug0_tunnel` 171 / `_aug0_sunlit` 87 / `_aug0_weather` 75 |
| 每张基底几个副本 | 1（399 张副本 → 399 个互不相同的基底） |
| 基底是否在基准内 | **399 / 399 全部在 534 张评测基准内** |
| 旧集真实构成 | 899 = **500 张原图 + 399 张增强副本**（此前盘点只数了顶层 500，漏了一半） |

## 结论

这 399 张**不是新数据**，且存在三重问题：

1. **泄漏**：基底图全部在测试基准中，同一场景同一缺陷的像素级近似副本进训练 = 直接泄漏。
2. **无新信息**：不增加任何新场景、新缺陷、新 GT。
3. **操作已被证伪**：它们是光照/低照度变体，与消融A 属同一类操作；
   消融A 在 8.31 报告 §13.4 被判为**确定性负结果**（fold2 宏AP −0.1203，
   等误报率下差 11.7 倍）。

## 正确做法

- 排除 `data/Defect_dataset/images/*_aug*_*.jpg` 及其标签。
- 使用 `scripts/check_leakage_landmines_20260901.py` 做数据构建期的泄漏体检：
    python scripts/check_leakage_landmines_20260901.py              # 全量体检
    python scripts/check_leakage_landmines_20260901.py --scan data/xxx
  或在代码里：
    from check_leakage_landmines_20260901 import LeakageGuard
    LeakageGuard().assert_clean(train_stems)

- 完整结论见：`docs/plans/9.01全量数据盘点/全量数据盘点与可靠结论_20260901.md` §4.1 / §4.2

## 副产物

本脚本曾生成 `data/Defect_dataset_unused399/`（manifest.csv / labels_16c / blind_test.txt）。
**该目录内容基于错误假设，不可使用**，保留仅为审计留痕。
"""

import sys

def main():
    print(__doc__)
    print("\n[已阻止执行] 本脚本基于错误假设，不做任何操作。")
    return 1

if __name__ == "__main__":
    sys.exit(main())
