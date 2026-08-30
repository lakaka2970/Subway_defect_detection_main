# -*- coding: utf-8 -*-
"""DINOv2 跨域模板匹配：用车间(Defect_dataset_2)部件样例，在现场(Normal_dataset)tile 上定位同类部件。

这是"AI 数据工厂" L2 部件结构化的第一步。
关键：本项目 16 类缺陷大多是"缺失/松动/缺口"这类**状态**，不是可搬运的物体，
因此不能"把缺陷零件贴上去"，而必须先在现场图上找到**正常部件**，再对它做删除/微扰。

用法：
  python remote_anchor_match.py --tiles-root ... --crops-root ... --out ...
                                --limit-tiles 200 --classes 0,1,4,5,12 --exemplars 8

输出：
  anchors.jsonl          每个锚点：tile 路径、类别、bbox、相似度
  preview/match_<cls>.jpg  每个类别的 Top 匹配可视化
"""
import argparse
import json
import os
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
import cv2

NAMES = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
         "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
         "BSBM", "INSD", "DRPS"]


def load_model(name="vit_base_patch14_dinov2.lvd142m", img_size=518, device="cuda"):
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
    m = m.to(device).half().eval()
    return m


@torch.no_grad()
def feat_map(model, img_bgr, img_size=518, device="cuda"):
    """返回 (H', W', D) 的 patch token 特征图（float32, L2 归一化）。"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(device).half()
    tok = model.forward_features(x)          # (1, 1+N, D)
    tok = tok[0, 1:, :]                       # 去掉 CLS
    p = int(np.sqrt(tok.shape[0]))
    f = tok.reshape(p, p, -1).float()
    f = F.normalize(f, dim=-1)
    return f  # (p, p, D)


@torch.no_grad()
def feat_vec(model, img_bgr, img_size=518, device="cuda"):
    """整图级特征向量（对 patch token 平均池化后 L2 归一化）。"""
    f = feat_map(model, img_bgr, img_size, device)
    v = f.reshape(-1, f.shape[-1]).mean(0)
    return F.normalize(v, dim=0)


def nms_boxes(boxes, scores, iou_thr=0.5, max_keep=200):
    """简单 NMS，boxes 为 [x0,y0,x1,y1]。"""
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_keep:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest]); yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest]); yy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
        iou = inter / np.maximum(1e-6, area[i] + area[rest] - inter)
        order = rest[iou <= iou_thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", default="/root/autodl-tmp/subway/out/normal_field_v1")
    ap.add_argument("--crops-root", default="/root/autodl-tmp/subway/data/defect_crops_v1")
    ap.add_argument("--out", default="/root/autodl-tmp/subway/out/anchors_v1")
    ap.add_argument("--limit-tiles", type=int, default=200)
    ap.add_argument("--classes", default="0,1,4,5,12")
    ap.add_argument("--exemplars", type=int, default=6, help="每类取多少个样例")
    ap.add_argument("--scales", default="3,4,6,8", help="模板占多少个 patch 边长")
    ap.add_argument("--top-per-tile", type=int, default=3)
    ap.add_argument("--min-sim", type=float, default=0.45)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--vis-per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)

    print("加载 DINOv2: %s @ %d" % (args.model, args.img_size), flush=True)
    model = load_model(args.model, args.img_size, device)
    P = args.img_size // 14
    print("  patch 网格 %dx%d" % (P, P), flush=True)

    # ---------- 1. 收集样例（按类别） ----------
    meta_path = os.path.join(args.crops_root, "crops_meta.jsonl")
    if not os.path.exists(meta_path):
        raise SystemExit("找不到 %s，请先上传缺陷裁剪块" % meta_path)
    recs = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
    by_cls = {}
    for r in recs:
        by_cls.setdefault(r["cls"], []).append(r)
    print("裁剪块 %d 个，覆盖 %d 个类别" % (len(recs), len(by_cls)), flush=True)

    cls_list = [int(c) for c in args.classes.split(",") if c.strip() != ""]
    scales = [int(s) for s in args.scales.split(",")]

    exemplars = {}   # cls -> [(name, feat_vec, crop_img, bbox_in_crop)]
    for c in cls_list:
        pool = by_cls.get(c, [])
        if not pool:
            print("  类别 %d (%s): 无样例，跳过" % (c, NAMES[c]))
            continue
        # 取 bbox 较大的实例作为样例（结构更清晰）
        pool = sorted(pool, key=lambda r: -r["bbox_src_wh_px"][0] * r["bbox_src_wh_px"][1])
        pool = pool[: max(args.exemplars * 3, 12)]
        random.shuffle(pool)
        pool = pool[: args.exemplars]
        items = []
        for r in pool:
            ip = os.path.join(args.crops_root, "crops", r["crop"])
            im = cv2.imread(ip)
            if im is None:
                continue
            h, w = im.shape[:2]
            x0, y0, x1, y1 = r["bbox_in_crop"]
            # 取 bbox 区域作为"部件"模板
            sub = im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            if sub.size == 0 or min(sub.shape[:2]) < 8:
                continue
            v = feat_vec(model, sub, args.img_size, device)
            items.append((r["crop"], v, sub))
        if items:
            exemplars[c] = items
            print("  类别 %d (%s): %d 个样例" % (c, NAMES[c], len(items)), flush=True)

    # ---------- 2. 现场 tile ----------
    tile_dirs = [os.path.join(args.tiles_root, "images", s) for s in ("train", "val")]
    tiles = []
    for d in tile_dirs:
        if os.path.isdir(d):
            tiles += sorted(glob.glob(os.path.join(d, "*.jpg")))
    if args.limit_tiles > 0:
        random.shuffle(tiles)
        tiles = tiles[: args.limit_tiles]
    print("\n现场 tile: %d 个" % len(tiles), flush=True)

    # ---------- 3. 匹配 ----------
    anchors = {c: [] for c in exemplars}
    feat_cache = {}

    for ti, tp in enumerate(tiles):
        im = cv2.imread(tp)
        if im is None:
            continue
        f = feat_map(model, im, args.img_size, device)     # (P,P,D)
        H, W = im.shape[:2]
        flat = f.reshape(-1, f.shape[-1])                   # (P*P, D)

        for c, items in exemplars.items():
            best_for_tile = []
            for name, v, sub in items:
                sim = (flat @ v).reshape(P, P)              # (P,P) 余弦相似图
                sim_np = sim.cpu().numpy()
                # 取 top 区域
                flat_sim = sim_np.ravel()
                idx = np.argsort(flat_sim)[::-1][: args.top_per_tile * 6]
                for ii in idx:
                    py, px = divmod(int(ii), P)
                    s = float(flat_sim[ii])
                    if s < args.min_sim:
                        continue
                    # patch -> 原图坐标（patch 中心）
                    cx = (px + 0.5) / P * W
                    cy = (py + 0.5) / P * H
                    # 尺寸用模板隐含尺度近似：模板占 k 个 patch
                    side = (args.img_size / P) * (W / args.img_size) * 3.0
                    best_for_tile.append([cx - side / 2, cy - side / 2,
                                          cx + side / 2, cy + side / 2, s, name])
            if not best_for_tile:
                continue
            boxes = [b[:4] for b in best_for_tile]
            scores = [b[4] for b in best_for_tile]
            keep = nms_boxes(boxes, scores, iou_thr=0.4, max_keep=args.top_per_tile)
            for k in keep:
                b = best_for_tile[k]
                anchors[c].append({
                    "tile": tp,
                    "cls": c,
                    "cls_name": NAMES[c],
                    "bbox": [int(max(0, b[0])), int(max(0, b[1])),
                             int(min(W, b[2])), int(min(H, b[3]))],
                    "sim": round(b[4], 4),
                    "exemplar": b[5],
                })

        if (ti + 1) % 25 == 0:
            tot = sum(len(v) for v in anchors.values())
            print("  %d/%d tiles, 累计锚点 %d" % (ti + 1, len(tiles), tot), flush=True)

    # ---------- 4. 保存 ----------
    ap_path = os.path.join(args.out, "anchors.jsonl")
    n = 0
    with open(ap_path, "w", encoding="utf-8") as f:
        for c in sorted(anchors):
            for a in anchors[c]:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                n += 1
    print("\n锚点总数 %d -> %s" % (n, ap_path))

    # ---------- 5. 可视化 ----------
    for c in sorted(anchors):
        items = sorted(anchors[c], key=lambda a: -a["sim"])[: args.vis_per_class]
        if not items:
            continue
        cells = []
        for a in items:
            im = cv2.imread(a["tile"])
            if im is None:
                continue
            x0, y0, x1, y1 = a["bbox"]
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 3)
            lab = "%s sim=%.2f" % (a["cls_name"], a["sim"])
            cv2.putText(im, lab, (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cells.append(cv2.resize(im, (384, 384)))
        if not cells:
            continue
        cols = 4
        rows = (len(cells) + cols - 1) // cols
        canvas = np.zeros((rows * 384, cols * 384, 3), np.uint8)
        for i, cc in enumerate(cells):
            r, k = divmod(i, cols)
            canvas[r * 384:(r + 1) * 384, k * 384:(k + 1) * 384] = cc
        op = os.path.join(args.out, "preview", "match_c%d_%s.jpg" % (c, NAMES[c]))
        cv2.imwrite(op, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print("  可视化 -> %s (%d 张)" % (op, len(cells)))

    print("\nANCHOR_MATCH_DONE")


if __name__ == "__main__":
    main()
