# -*- coding: utf-8 -*-
"""把 Defect_dataset_2 中每个标注缺陷实例裁成小图导出，供远端 SAM 抠图使用。

只上传裁剪块（约 1GB）而不是 31GB 全量原图。
裁剪策略：
  - 以 bbox 为中心，按 pad 倍率外扩（默认 2.2x）
  - 最长边 cap 到 --cap 像素（默认 640）
  - 保证裁剪块内的 bbox 像素尺寸 >= 该实例在 1280 滑窗训练分辨率下的原生尺寸
  - JPEG q95 保存，附带 bbox 在裁剪块内的相对坐标

输出：
  <out>/crops/<stem>__i<idx>__c<cls>.jpg
  <out>/crops_meta.jsonl   每行一个实例的完整元数据
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ROOT = r"E:\Work\Subway_defect_detection_main"


def read_classes(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def parse_label(path, n2i):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.split()
            if len(t) < 5:
                continue
            try:
                c = int(t[0]) if t[0].isdigit() else n2i[t[0]]
                xc, yc, w, h = (float(v) for v in t[1:5])
            except Exception:
                continue
            out.append((c, xc, yc, w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=os.path.join(ROOT, "data", "Defect_dataset_2",
                                                      "Defect_dataset", "images"))
    ap.add_argument("--lbl-dir", default=os.path.join(ROOT, "data", "Defect_dataset_2",
                                                      "Defect_dataset", "labels"))
    ap.add_argument("--classes", default=os.path.join(ROOT, "data", "train_data_2", "classes.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "defect_crops_v1"))
    ap.add_argument("--pad", type=float, default=2.2, help="bbox 外扩倍率")
    ap.add_argument("--cap", type=int, default=640, help="裁剪块最长边上限(px)")
    ap.add_argument("--min-side", type=int, default=48, help="小于此边长的实例跳过")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张图（调试用）")
    args = ap.parse_args()

    names = read_classes(args.classes)
    n2i = {n: i for i, n in enumerate(names)}
    crop_dir = os.path.join(args.out, "crops")
    os.makedirs(crop_dir, exist_ok=True)

    lfs = sorted(glob.glob(os.path.join(args.lbl_dir, "*.txt")))
    if args.limit:
        lfs = lfs[: args.limit]

    meta_path = os.path.join(args.out, "crops_meta.jsonl")
    n_out = 0
    n_skip_small = 0
    n_no_img = 0
    total_bytes = 0

    with open(meta_path, "w", encoding="utf-8") as mf:
        for li, lf in enumerate(lfs):
            stem = os.path.splitext(os.path.basename(lf))[0]
            ip = os.path.join(args.img_dir, stem + ".jpg")
            if not os.path.exists(ip):
                # 容忍 png
                for ext in (".png", ".JPG", ".jpeg"):
                    alt = os.path.join(args.img_dir, stem + ext)
                    if os.path.exists(alt):
                        ip = alt
                        break
            if not os.path.exists(ip):
                n_no_img += 1
                continue

            rows = parse_label(lf, n2i)
            if not rows:
                continue

            im = cv2.imread(ip)
            if im is None:
                n_no_img += 1
                continue
            H, W = im.shape[:2]

            for idx, (c, xc, yc, bw, bh) in enumerate(rows):
                # bbox 绝对像素
                bx0 = (xc - bw / 2) * W
                by0 = (yc - bh / 2) * H
                bx1 = (xc + bw / 2) * W
                by1 = (yc + bh / 2) * H
                bwid = bx1 - bx0
                bhei = by1 - by0

                # 原生尺寸（该实例在 1280 训练分辨率下的像素尺寸）
                native_at_1280 = bwid * 1280.0 / W

                # 外扩
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                cw, ch = bwid * args.pad, bhei * args.pad
                x0 = int(round(cx - cw / 2))
                y0 = int(round(cy - ch / 2))
                x1 = int(round(cx + cw / 2))
                y1 = int(round(cy + ch / 2))

                # 缩放到 cap
                scale = 1.0
                if max(x1 - x0, y1 - y0) > args.cap:
                    scale = args.cap / float(max(x1 - x0, y1 - y0))
                if scale < 1.0:
                    # 先按 scale 重采样整图再裁，等价于裁剪后缩放但避免边界问题
                    ims = cv2.resize(im, (int(W * scale), int(H * scale)),
                                     interpolation=cv2.INTER_AREA)
                    sx0, sy0 = int(x0 * scale), int(y0 * scale)
                    sx1, sy1 = int(x1 * scale), int(y1 * scale)
                else:
                    ims = im
                    sx0, sy0, sx1, sy1 = x0, y0, x1, y1

                # 裁剪（带 padding 补齐，避免越界后尺寸变小导致 bbox 相对坐标漂移）
                ch_h, ch_w = ims.shape[:2]
                pad_l = max(0, -sx0)
                pad_t = max(0, -sy0)
                pad_r = max(0, sx1 - ch_w)
                pad_b = max(0, sy1 - ch_h)
                cx0, cy0 = max(0, sx0), max(0, sy0)
                cx1, cy1 = min(ch_w, sx1), min(ch_h, sy1)
                patch = ims[cy0:cy1, cx0:cx1]
                if patch.size == 0:
                    n_skip_small += 1
                    continue
                if any([pad_l, pad_t, pad_r, pad_b]):
                    patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r,
                                               cv2.BORDER_REPLICATE)
                ph, pw = patch.shape[:2]
                if min(ph, pw) < args.min_side:
                    n_skip_small += 1
                    continue

                # bbox 在裁剪块内的相对坐标
                fx0 = (bx0 * scale - sx0 + pad_l) / pw
                fy0 = (by0 * scale - sy0 + pad_t) / ph
                fx1 = (bx1 * scale - sx0 + pad_l) / pw
                fy1 = (by1 * scale - sy0 + pad_t) / ph

                fn = "%s__i%d__c%d.jpg" % (stem, idx, c)
                op = os.path.join(crop_dir, fn)
                ok = cv2.imwrite(op, patch, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                if not ok:
                    continue

                rec = {
                    "crop": fn,
                    "src_image": stem + ".jpg",
                    "src_size": [W, H],
                    "cls": c,
                    "cls_name": names[c],
                    "inst_idx": idx,
                    "bbox_in_crop": [round(fx0, 6), round(fy0, 6), round(fx1, 6), round(fy1, 6)],
                    "bbox_src_px": [round(bx0, 1), round(by0, 1), round(bx1, 1), round(by1, 1)],
                    "bbox_src_wh_px": [round(bwid, 1), round(bhei, 1)],
                    "native_side_at_1280": round(max(native_at_1280,
                                                     bhei * 1280.0 / H), 1),
                    "crop_size": [pw, ph],
                    "scale": round(scale, 4),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_bytes += os.path.getsize(op)
                n_out += 1

            if (li + 1) % 500 == 0:
                print("  %d/%d 张, 已导出 %d 个裁剪块, %.0f MB"
                      % (li + 1, len(lfs), n_out, total_bytes / 1e6), flush=True)

    print()
    print("完成：导出 %d 个实例裁剪块 -> %s" % (n_out, crop_dir))
    print("  跳过(过小) %d, 缺图 %d" % (n_skip_small, n_no_img))
    print("  总体积 %.0f MB，平均 %.0f KB/块" % (total_bytes / 1e6, total_bytes / max(1, n_out) / 1024))
    print("  元数据 -> %s" % meta_path)


if __name__ == "__main__":
    main()
