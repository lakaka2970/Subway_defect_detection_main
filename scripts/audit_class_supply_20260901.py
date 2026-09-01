# -*- coding: utf-8 -*-
"""
细分类别扩展 —— 供数能力审计（2026-09-01）

目的：评估"16 大类 -> 30~40 小类"时，两条技术路线的样本供给能力。
核心要回答一个问题：样本需求是随"状态数 x 部位数"增长（路径1），
还是随"状态数 + 部位数"增长（路径2）。

统计口径：
  - 规范 taxonomy（DEFECT_CLASSES 顺序）= 中文对照表口径
  - 模型顺序（MODEL_CLASS_ORDER）= 磁盘标签文件实际使用的编号
本脚本只用"英文编码 + 中文名"，不依赖编号，避免两套口径打架。
"""
import os, json, collections, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 规范 taxonomy（docs/模型输出与缺陷类型对照表.md 第一节）
CANON = [
    ("VHBNM",   "垂直悬吊安装底座螺母缺失",     "刚性", "serious"),
    ("VHBNL",   "垂直悬吊安装底座螺母松动",     "刚性", "serious"),
    ("SVHBNM",  "单支垂直悬吊槽钢底座螺母缺失", "刚性", "serious"),
    ("SVHBNL",  "单支垂直悬吊槽钢底座螺母松动", "刚性", "serious"),
    ("SVHTNL",  "单支垂直悬吊槽钢上方螺母松动", "刚性", "normal"),
    ("RHTBNM",  "刚性悬挂吊柱底座螺母缺失",     "刚性", "serious"),
    ("RHTBNL",  "刚性悬挂吊柱底座螺母松动",     "刚性", "serious"),
    ("GWCSBNM", "地线线夹托板安装底座螺母缺失", "刚性", "serious"),
    ("GWCSBNL", "地线线夹托板安装底座螺母松动", "刚性", "serious"),
    ("GWCNM",   "地线线夹螺母缺失",             "刚性", "serious"),
    ("GWCNL",   "地线线夹螺母松动",             "刚性", "serious"),
    ("BSBM",    "汇流排中间接头螺栓缺失",       "刚性", "critical"),
    ("INSD",    "绝缘子破损",                   "刚性", "critical"),
    ("CBHPM",   "腕臂底座横向销钉缺失",         "柔性", "serious"),
    ("CBVPM",   "腕臂底座垂直销钉缺失",         "柔性", "serious"),
    ("DRPS",    "吊弦不受力",                   "柔性", "serious"),
]
CANON_ID = {c[0]: i for i, c in enumerate(CANON)}
CN = {c[0]: c[1] for c in CANON}
SYS = {c[0]: c[2] for c in CANON}

# 模型顺序（train_data_2/classes.txt）：label 文件里的编号 -> 编码
MODEL_ORDER = [
    "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL",
    "CBHPM", "CBVPM", "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL",
    "GWCNM", "GWCNL", "BSBM", "INSD", "DRPS",
]

# ---------- "部位 x 状态" 因子拆解 ----------
# 从规范命名里抽出的两个正交因子
PART = {  # 部位（部件）
    "VHBNM": "垂直悬吊安装底座", "VHBNL": "垂直悬吊安装底座",
    "SVHBNM": "垂直悬吊槽钢底座", "SVHBNL": "垂直悬吊槽钢底座",
    "SVHTNL": "垂直悬吊槽钢上方",
    "RHTBNM": "刚性悬挂吊柱底座", "RHTBNL": "刚性悬挂吊柱底座",
    "GWCSBNM": "地线线夹托板安装底座", "GWCSBNL": "地线线夹托板安装底座",
    "GWCNM": "地线线夹", "GWCNL": "地线线夹",
    "BSBM": "汇流排中间接头",
    "INSD": "绝缘子",
    "CBHPM": "腕臂底座", "CBVPM": "腕臂底座",
    "DRPS": "吊弦",
}
STATE = {  # 状态原语
    "VHBNM": "螺母缺失", "VHBNL": "螺母松动",
    "SVHBNM": "螺母缺失", "SVHBNL": "螺母松动",
    "SVHTNL": "螺母松动",
    "RHTBNM": "螺母缺失", "RHTBNL": "螺母松动",
    "GWCSBNM": "螺母缺失", "GWCSBNL": "螺母松动",
    "GWCNM": "螺母缺失", "GWCNL": "螺母松动",
    "BSBM": "螺栓缺失",
    "INSD": "破损",
    "CBHPM": "销钉缺失", "CBVPM": "销钉缺失",
    "DRPS": "不受力",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def label_dir_for(img_dir: Path):
    """按常见约定找标签目录"""
    for cand in [
        img_dir.parent / "labels" / img_dir.name,
        img_dir.parent / "labels",
        img_dir / "labels",
    ]:
        if cand.is_dir():
            return cand
    return None


def scan(img_dir: Path, recursive=True):
    """统计一个图像目录下的类别实例数（标签用模型顺序编号）"""
    if not img_dir or not img_dir.is_dir():
        return collections.Counter(), 0, 0
    ldir = label_dir_for(img_dir)
    if ldir is None:
        # 尝试同名的 labels 兄弟
        p = img_dir.parent / "labels"
        ldir = p if p.is_dir() else None
    if ldir is None:
        return collections.Counter(), 0, 0

    hist = collections.Counter()
    n_img = 0
    n_box = 0
    it = img_dir.rglob("*") if recursive else img_dir.glob("*")
    for f in it:
        if f.suffix.lower() not in IMG_EXT:
            continue
        n_img += 1
        lp = ldir / (f.stem + ".txt")
        # 递归模式下标签可能也在子目录
        if not lp.exists():
            lp = ldir / f.relative_to(img_dir).with_suffix(".txt")
        if not lp.exists():
            continue
        try:
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                if 0 <= cid < len(MODEL_ORDER):
                    hist[MODEL_ORDER[cid]] += 1
                    n_box += 1
        except Exception:
            pass
    return hist, n_img, n_box


def main():
    targets = [
        ("Defect_dataset_2(车间)", ROOT / "data/Defect_dataset_2/Defect_dataset/images", True),
        ("train_data_2(训练清单)", ROOT / "data/train_data_2/images", True),
        ("Defect_dataset_16_rebuilt(目标域基准)", ROOT / "data/Defect_dataset_16_rebuilt/images", True),
        ("Defect_dataset(旧7类)", ROOT / "data/Defect_dataset/images", True),
    ]

    out = {}
    print("=" * 96)
    print("一、各数据源的 16 类实例供给（标签按模型顺序编号）")
    print("=" * 96)
    hdr = f"{'编码':<9}{'中文名':<22}{'系统':<5}" + "".join(f"{n.split('(')[0]:>12}" for n, _, _ in targets)
    print(hdr)
    print("-" * 96)
    for code, cname, sysname, _sev in CANON:
        row = f"{code:<9}{cname:<20}{sysname:<5}"
        for name, p, rec in targets:
            h = out.setdefault(name, scan(p, rec))
            row += f"{h[0].get(code, 0):>12}"
        print(row)
    print("-" * 96)
    row = f"{'合计':<9}{'':<20}{'':<5}"
    for name, p, rec in targets:
        h = out[name]
        row += f"{sum(h[0].values()):>12}"
    print(row)

    print()
    print("=" * 96)
    print("二、因子拆解：现有 16 类 = 部位 x 状态 的笛卡尔积（未显式解耦）")
    print("=" * 96)
    parts = sorted(set(PART.values()))
    states = sorted(set(STATE.values()))
    print(f"部位因子（{len(parts)} 个）：{', '.join(parts)}")
    print(f"状态因子（{len(states)} 个）：{', '.join(states)}")
    print(f"理论满秩组合 = {len(parts)} x {len(states)} = {len(parts)*len(states)}")
    exist = set((PART[c], STATE[c]) for c in CANON_ID)
    print(f"现有 16 类实际占用组合 = {len(exist)}  ->  稀疏度 {len(exist)/(len(parts)*len(states)):.1%}")

    print()
    print("=" * 96)
    print("三、路径1 vs 路径2 的样本需求算术")
    print("=" * 96)
    # 用户预估 30-40 小类
    for n_sub in (30, 36, 40):
        print(f"\n  目标小类数 = {n_sub}")
        print(f"    路径1（端到端，每小类独立学习）：需要 {n_sub} 组独立缺陷样本")
        # 路径2：状态头 + 部位头
        # 估计：小类数 ~= 有效部位数 x 有效状态数，取接近 n_sub 的因子分解
        best = None
        for np_ in range(3, 13):
            ns = -(-n_sub // np_)
            cost = np_ + ns
            if best is None or cost < best[0]:
                best = (cost, np_, ns)
        cost, np_, ns = best
        print(f"    路径2（状态头 + 部位头）：最省分解 ≈ {np_} 部位 + {ns} 状态 = {cost} 组样本")
        print(f"    -> 样本组数需求比 = {cost}/{n_sub} = {cost/n_sub:.0%}（省 {1-cost/n_sub:.0%}）")

    print()
    print("=" * 96)
    print("四、关键：部位头的样本来自'正常部件'，与缺陷无关 -> 供给不受限")
    print("=" * 96)
    # Normal_dataset / tiles_normal 里的正常部件数量
    nd = ROOT / "data/Normal_dataset"
    n_img = 0
    if nd.is_dir():
        for f in nd.rglob("*"):
            if f.suffix.lower() in IMG_EXT:
                n_img += 1
    print(f"  Normal_dataset（现场无缺陷图，全部可用作部件头样本源）：{n_img} 张 5120x5120 原图")
    tiles_normal = len(list((ROOT / "data/tiles_normal").rglob("*.jpg"))) if (ROOT / "data/tiles_normal").is_dir() else 0
    print(f"  tiles_normal（已切片）：{tiles_normal} 个 tile")
    n2 = ROOT / "data/Defect_dataset_2/Defect_dataset"
    if n2.is_dir():
        imgs = [f for f in (n2 / "images").rglob("*") if f.suffix.lower() in IMG_EXT] if (n2 / "images").is_dir() else []
        print(f"  Defect_dataset_2（车间 16 类，同一张图内既有缺陷也有大量正常部件）：{len(imgs)} 张")

    # 持久化
    payload = {
        "per_source": {k: {"hist": dict(v[0]), "n_img": v[1], "n_box": v[2]} for k, v in out.items()},
        "n_parts": len(parts),
        "n_states": len(states),
        "parts": parts,
        "states": states,
        "existing_combos": sorted(list(exist)),
        "normal_images": n_img,
        "tiles_normal": tiles_normal,
    }
    op = ROOT / "docs/plans/9.01全量数据盘点/class_supply_audit.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 已保存 {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
