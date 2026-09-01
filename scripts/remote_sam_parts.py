# -*- coding: utf-8 -*-
"""SAM 自动分割 + DINOv2 图像级分类：在现场 tile 上定位可编辑部件。

流程：
  1. SAM(everything 模式) 对现场 tile 分割，得到所有候选掩码
  2. 按面积过滤（螺母/螺栓/销钉这类紧固件是中小尺寸目标）
  3. 每个候选裁剪后用 DINOv2 提特征，与车间样例库(按 16 类)比对
  4. 输出部件清单 + 可视化（掩码叠加 + 类别标签）

用法：
  python remote_sam_parts.py --tiles-root ... --crops-root ... --out ... --limit-tiles 50
"""
import argparse
import collections
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import cv2

NAMES = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
         "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
         "BSBM", "INSD", "DRPS"]

# 1280 tile 上紧固件的合理面积范围（px^2）
AREA_MIN, AREA_MAX = 120, 60000


def load_dino(name="vit_base_patch14_dinov2.lvd142m", img_size=518, device="cuda"):
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
    return m.to(device).half().eval()


@torch.no_grad()
def dino_vec(model, img_bgr, img_size=518, device="cuda"):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = ((x - 0.5) / 0.5).unsqueeze(0).to(device).half()
    tok = model.forward_features(x)[0, 1:, :]
    v = tok.float().mean(0)
    return F.normalize(v, dim=0)


def build_exemplar_lib(model, crops_root, per_class=12, device="cuda", img_size=518):
    """从车间裁剪块构建每类的 DINOv2 特征库（只取 bbox 内区域）。"""
    meta_path = os.path.join(crops_root, "crops_meta.jsonl")
    recs = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
    by_cls = collections.defaultdict(list)
    for r in recs:
        by_cls[r["cls"]].append(r)
    lib = {}
    for c, pool in sorted(by_cls.items()):
        # 结构清晰的大实例优先，随机取样避免同图偏置
        pool = sorted(pool, key=lambda r: -(r["bbox_src_wh_px"][0] * r["bbox_src_wh_px"][1]))
        pool = pool[: per_class * 4]
        random.shuffle(pool)
        vecs = []
        for r in pool[:per_class]:
            ip = os.path.join(crops_root, "crops", r["crop"])
            im = cv2.imread(ip)
            if im is None:
                continue
            h, w = im.shape[:2]
            x0, y0, x1, y1 = r["bbox_in_crop"]
            sub = im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            if sub.size == 0 or min(sub.shape[:2]) < 8:
                continue
            vecs.append(dino_vec(model, sub, img_size, device))
        if vecs:
            lib[c] = torch.stack(vecs)          # (n, D)
            print("  样例库 %-8s %d 个" % (NAMES[c], len(vecs)), flush=True)
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", default="/root/autodl-tmp/subway/out/normal_field_v1")
    ap.add_argument("--crops-root", default="/root/autodl-tmp/subway/data/defect_crops_v1")
    ap.add_argument("--out", default="/root/autodl-tmp/subway/out/sam_parts_pilot")
    ap.add_argument("--sam-weights", default="sam2.1_b.pt")
    ap.add_argument("--limit-tiles", type=int, default=50)
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--min-sim", type=float, default=0.55)
    ap.add_argument("--vis", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)

    print("加载 DINOv2 ...", flush=True)
    dino = load_dino(img_size=args.img_size, device=device)
    print("构建车间样例特征库 ...", flush=True)
    lib = build_exemplar_lib(dino, args.crops_root, args.per_class, device, args.img_size)
    lib_mat = {c: v for c, v in lib.items()}

    print("加载 SAM: %s" % args.sam_weights, flush=True)
    from ultralytics import SAM
    sam = SAM(args.sam_weights)

    tiles = []
    for s in ("train", "val"):
        d = os.path.join(args.tiles_root, "images", s)
        if os.path.isdir(d):
            tiles += sorted(glob.glob(os.path.join(d, "*.jpg")))
    random.shuffle(tiles)
    tiles = tiles[: args.limit_tiles]
    print("tile: %d 个" % len(tiles), flush=True)

    parts = []
    stats = collections.Counter()
    for ti, tp in enumerate(tiles):
        im = cv2.imread(tp)
        if im is None:
            continue
        H, W = im.shape[:2]
        try:
            res = sam(tp, device=device, verbose=False)[0]
        except Exception as e:
            print("  SAM 失败 %s: %s" % (tp, e))
            continue
        if res.masks is None:
            stats["no_mask"] += 1
            continue
        masks = res.masks.data.cpu().numpy()          # (n, h, w) 与原图同分辨率或缩放
        mh, mw = masks.shape[1:]
        n = masks.shape[0]
        stats["masks"] += n

        # DINOv2 批量分类
        cands = []
        for i in range(n):
            m = (masks[i] > 0.5).astype(np.uint8)
            if mh != H or mw != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            a = int(m.sum())
            if a < AREA_MIN or a > AREA_MAX:
                stats["area_filtered"] += 1
                continue
            ys, xs = np.where(m > 0)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            # 用掩码裁剪 + 白底，突出部件本体
            sub = im[y0:y1 + 1, x0:x1 + 1].copy()
            sm = m[y0:y1 + 1, x0:x1 + 1]
            if sub.size == 0:
                continue
            bg = np.full_like(sub, 114)
            bg[sm > 0] = sub[sm > 0]
            v = dino_vec(dino, bg, args.img_size, device)
            cands.append((v, (int(x0), int(y0), int(x1 + 1), int(y1 + 1)), a, i, sm))

        if not cands:
            stats["no_cand"] += 1
            continue
        V = torch.stack([c[0] for c in cands])         # (n, D)
        for c, libv in lib_mat.items():                 # 逐类算最大相似
            sims = (V @ libv.T).max(dim=1).values.cpu().numpy()   # (n,)
            for k, s in enumerate(sims):
                if s < args.min_sim:
                    continue
                v, box, area, midx, sm = cands[k]
                parts.append({
                    "tile": tp, "cls": c, "cls_name": NAMES[c], "sim": round(float(s), 4),
                    "bbox": list(box), "area": area, "mask_idx": int(midx),
                })
                stats["kept"] += 1

        if (ti + 1) % 10 == 0:
            print("  %d/%d tiles, 部件候选 %d" % (ti + 1, len(tiles), len(parts)), flush=True)

    print("\n统计:", dict(stats), flush=True)

    parts.sort(key=lambda p: -p["sim"])
    with open(os.path.join(args.out, "parts.jsonl"), "w", encoding="utf-8") as f:
        for p in parts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("部件 %d 个 -> parts.jsonl" % len(parts))

    # 可视化：每类取 sim 最高的若干个
    by_cls = collections.defaultdict(list)
    for p in parts:
        by_cls[p["cls"]].append(p)
    for c, ps in sorted(by_cls.items()):
        ps = ps[: args.vis]
        cells = []
        for p in ps:
            im = cv2.imread(p["tile"])
            if im is None:
                continue
            x0, y0, x1, y1 = p["bbox"]
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(im, "%s %.2f" % (p["cls_name"], p["sim"]),
                        (x0, max(22, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cells.append(cv2.resize(im, (384, 384)))
        if not cells:
            continue
        cols = 4
        rows = (len(cells) + cols - 1) // cols
        canvas = np.zeros((rows * 384, cols * 384, 3), np.uint8)
        for i, cc in enumerate(cells):
            r, k = divmod(i, cols)
            canvas[r * 384:(r + 1) * 384, k * 384:(k + 1) * 384] = cc
        op = os.path.join(args.out, "preview", "parts_c%d_%s.jpg" % (c, NAMES[c]))
        cv2.imwrite(op, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print("  可视化 -> %s (%d)" % (op, len(cells)))

    print("\nSAM_PARTS_DONE")


if __name__ == "__main__":
    main()
