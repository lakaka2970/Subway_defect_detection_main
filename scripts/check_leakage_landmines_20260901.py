#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
泄漏地雷检查器（2026-09-01）

用途：在数据构建阶段扫一遍候选训练集，凡是图像 stem（含剥离 `_augN_*` 后缀后的基底）
      命中 534 张评测基准的，一律拒绝并输出名单。

背景：全量审计发现三处与评测基准重叠的数据源，其中两处是历史遗留目录，随时可能被
      第 3 轮数据脚本扫到：
        - data/Defect_dataset 的 399 张离线增强副本（基底 100% 在基准内）
        - data/train_data_3_raw        （含 313 张基准图）
        - data/train_data_3_raw_2560   （含 245 张基准图）

用法：
    python scripts/check_leakage_landmines_20260901.py                # 全量体检
    python scripts/check_leakage_landmines_20260901.py --scan <目录>   # 检查某个候选训练集
    python scripts/check_leakage_landmines_20260901.py --scan <目录> --write-blacklist

作为库使用：
    from check_leakage_landmines_20260901 import LeakageGuard
    guard = LeakageGuard()
    bad = guard.filter(train_stems)      # 返回命中的 stem
    ok  = guard.sanitize(train_stems)    # 返回干净名单
"""
import re, sys, json, argparse
from pathlib import Path
from collections import Counter

ROOT = Path(r"E:\Work\Subway_defect_detection_main")
BENCH = ROOT / "data" / "Defect_dataset_16_rebuilt"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUG_RE = re.compile(r"_aug\d+_\w+$")

# 已知的历史遗留地雷（相对 ROOT 的目录 → 说明）
KNOWN_LANDMINES = {
    "data/Defect_dataset": "899 张 = 500 原图(已进基准) + 399 张离线增强副本(基底 100% 在基准内)",
    "data/train_data_3_raw": "含 313 张基准图（8.13 混合域训练遗留，已废弃）",
    "data/train_data_3_raw_2560": "含 245 张基准图（同上，2560 版本）",
}


def base_of(stem: str) -> str:
    """剥离离线增强后缀：xxx_aug0_tunnel -> xxx"""
    return AUG_RE.sub("", stem)


class LeakageGuard:
    def __init__(self, bench_dir: Path = None):
        bench_dir = Path(bench_dir) if bench_dir else BENCH
        self.bench = set()
        for f in (bench_dir / "images").glob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                self.bench.add(f.stem)
        # 同时把基底也纳入（防止有人用 base 图本身）
        self.bench |= {base_of(s) for s in self.bench}
        if not self.bench:
            raise RuntimeError(f"基准目录为空或不存在: {bench_dir/'images'}")

    def hits(self, stems):
        """返回命中的 stem 列表（同时匹配原名与剥离后缀后的基底名）"""
        out = []
        for s in stems:
            if s in self.bench or base_of(s) in self.bench:
                out.append(s)
        return out

    def sanitize(self, stems):
        bad = set(self.hits(stems))
        return [s for s in stems if s not in bad]

    def check_dir(self, d: Path):
        """递归扫描目录，返回 (总图数, 命中数, 命中名单)"""
        d = Path(d)
        stems = {f.stem for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT}
        hits = self.hits(stems)
        return len(stems), len(hits), sorted(hits)

    def assert_clean(self, stems, name="训练集"):
        bad = self.hits(stems)
        if bad:
            raise AssertionError(
                f"[泄漏] {name} 中 {len(bad)} 张图像命中评测基准，例如: {bad[:5]}")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", help="检查指定候选训练集目录")
    ap.add_argument("--write-blacklist", action="store_true", help="把命中名单写到 data/leakage_blacklist.txt")
    args = ap.parse_args()

    guard = LeakageGuard()
    print("=" * 78)
    print(f"评测基准: {len(guard.bench)} 张（含基底展开）")
    print("=" * 78)

    if args.scan:
        n, nh, hits = guard.check_dir(ROOT / args.scan if not Path(args.scan).is_absolute() else args.scan)
        print(f"\n[扫描] {args.scan}: {n} 张图 | 命中基准 {nh} 张")
        if hits:
            print(f"  示例: {hits[:5]}")
            if args.write_blacklist:
                p = ROOT / "data" / "leakage_blacklist.txt"
                p.write_text("\n".join(hits), encoding="utf-8")
                print(f"  黑名单已写入: {p}")
        else:
            print("  ✅ 干净，可直接进训练")
        return

    # 全量体检
    print(f"\n{'数据源':<38}{'图像':>8}{'命中基准':>10}{'剥离aug后命中':>14}  判定")
    print("-" * 78)
    report = {}
    for rel, note in KNOWN_LANDMINES.items():
        d = ROOT / rel
        if not d.exists():
            print(f"{rel:<38}{'—':>8}{'—':>10}{'—':>14}  目录不存在")
            continue
        n, nh, hits = guard.check_dir(d)
        report[rel] = {"n_images": n, "n_hits": nh, "note": note, "examples": hits[:5]}
        flag = "🔴 泄漏源" if nh else "✅ 干净"
        print(f"{rel:<38}{n:>8}{nh:>10}{'':>14}  {flag}")

    # 当前在用的训练集与切片池（应为干净）
    clean = {
        "data/train_data_2（当前训练集）": ROOT / "data" / "train_data_2",
        "data/tiles_normal（负样本切片）": ROOT / "data" / "tiles_normal",
        "data/tiles_workshop（车间切片）": ROOT / "data" / "tiles_workshop",
    }
    print("-" * 78)
    for name, d in clean.items():
        if not d.exists():
            continue
        n, nh, hits = guard.check_dir(d)
        # tiles_* 的文件名形如 R_xxx_t3 / W_xxx_t5，需特殊处理
        if d.name.startswith("tiles_"):
            srcs = {s.split("_t")[0].lstrip("RW") for s in
                    {f.stem for f in (d / "images").glob("*") if f.suffix.lower() in IMG_EXT}}
            nh = len(guard.hits(srcs))
            n = len(srcs)
        report[name] = {"n_source_images": n, "n_hits": nh}
        flag = "🔴 泄漏" if nh else "✅ 干净"
        print(f"{name:<38}{n:>8}{nh:>10}{'':>14}  {flag}")

    print("-" * 78)
    print("\n说明：`data/Defect_dataset` 的 899 张 = 500 原图（已进基准）+ 399 张离线增强副本。")
    print("      副本文件名形如 xxx_aug0_tunnel，剥离后缀后基底 100% 在基准内，")
    print("      且属项目已证伪的操作（与消融A 同为光照/低照度变体）。建议整体排除。")

    out = ROOT / "docs" / "plans" / "9.01全量数据盘点" / "leakage_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整报告: {out}")


if __name__ == "__main__":
    main()
