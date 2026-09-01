# -*- coding: utf-8 -*-
"""交付候选模型数据准备：全部 931 张 Normal 切片 + 全量训练清单。

deploy_train = W base 14,660 + N 全部（220 已切 + 711 新切，约 23k）
             + R 全部 534 图 13,350 + 207 个硬负样本 tile 过采样 x3
deploy_val   = V 3,091 + 全部监控折真实 tile（仅收敛 sanity，不做选择依据）
说明：交付模型无留出折，其指标以四折 LOSO 模型为代表；本模型用于 10/15 部署。
"""
import csv
import glob
import json
import os
import sys
import tarfile
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_phase1_tiles_20260830 as bt

IDX = os.path.join(ROOT, "data", "tiles_index")
OUT_N = os.path.join(ROOT, "data", "tiles_normal")
RROOT = "/root/autodl-tmp/subway/data/tiles"


def main():
    os.makedirs(os.path.join(OUT_N, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_N, "labels"), exist_ok=True)

    bench = {os.path.basename(p) for p in glob.glob(
        os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt",
                     "images", "*.jpg"))}
    conflict = {r["image"] for r in csv.DictReader(open(
        os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt",
                     "rebuild_audit.csv"), encoding="utf-8-sig"))
        if r["source"] == "Normal_dataset" and r["status"] == "MISMATCH"}
    used = bench | conflict
    have = {os.path.splitext(p)[0][2:].rsplit("_t", 1)[0] + ".jpg"
            for p in glob.glob(os.path.join(OUT_N, "images", "N_*.jpg"))}
    todo = []
    for p in glob.glob(os.path.join(ROOT, "data", "Normal_dataset", "images",
                                    "**", "*.jpg"), recursive=True):
        b = os.path.basename(p)
        if b not in used and b not in have:
            todo.append((p, "", os.path.splitext(b)[0], "N", OUT_N))
    print("待切 Normal:", len(todo))
    if todo:
        with Pool(8) as pool:
            res = pool.map(bt.tile_full, todo)
        print("新 tile:", sum(r[1] for r in res))

    w = ["%s/workshop/images/%s.jpg" % (RROOT, os.path.splitext(f)[0])
         for f in sorted(os.path.basename(p) for p in glob.glob(
             os.path.join(ROOT, "data", "tiles_workshop", "images",
                          "W_IMG_*.jpg")))]
    # base 过滤（排除增强源）
    man = json.load(open(os.path.join(ROOT, "data", "train_data_2",
                                      "manifest.json"), encoding="utf-8"))
    aug_out = {v["out"] for v in man["variants"]}
    w = [p for p in w
         if os.path.splitext(os.path.basename(p))[0][2:].rsplit(
             "_t", 1)[0] not in aug_out]
    n = sorted("%s/normal/images/%s" % (RROOT, os.path.basename(p))
               for p in glob.glob(os.path.join(OUT_N, "images", "N_*.jpg")))
    r = sorted("%s/real/images/%s" % (RROOT, os.path.basename(p))
               for p in glob.glob(os.path.join(ROOT, "data", "tiles_real",
                                               "images", "R_*.jpg")))
    hn = []
    seen = set()
    for ln in open(os.path.join(ROOT, "docs", "plans", "8.31阶段1产物",
                                "hn_boxes_round1.jsonl"), encoding="utf-8"):
        b = json.loads(ln)
        stem = os.path.splitext(b["image"])[0]
        cx = (b["box"][0] + b["box"][2]) / 2
        cy = (b["box"][1] + b["box"][3]) / 2
        tn = "R_%s_t%d" % (stem, min(int(cy / 960), 4) * 5 +
                           min(int(cx / 960), 4))
        p = "%s/real/images/%s.jpg" % (RROOT, tn)
        if p not in seen:
            seen.add(p)
            hn += [p] * 3
    with open(os.path.join(IDX, "deploy_train.txt"), "w",
              encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(w + n + r + hn))
    print("deploy_train: W %d + N %d + R %d + HN %d = %d" %
          (len(w), len(n), len(r), len(hn), len(w) + len(n) + len(r) + len(hn)))

    v = sorted("%s/workshop_val/images/%s" % (RROOT, os.path.basename(p))
               for p in glob.glob(os.path.join(ROOT, "data",
                                               "tiles_workshop_val",
                                               "images", "V_*.jpg")))
    mon = []
    for k in range(4):
        for ln in open(os.path.join(IDX, "val_fold%d.txt" % k),
                       encoding="utf-8"):
            if "/real/images/" in ln:
                mon.append(ln.strip())
    with open(os.path.join(IDX, "deploy_val.txt"), "w", encoding="utf-8",
              newline="\n") as fp:
        fp.write("\n".join(v + sorted(set(mon))))
    print("deploy_val: V %d + mon %d" % (len(v), len(set(mon))))

    new_tiles = [os.path.splitext(os.path.basename(p))[0]
                 for p, _, _, _, _ in todo for q in [None]]
    out = os.path.join(ROOT, "data", "_upload", "normal_extra_tiles.tar")
    cnt = 0
    with tarfile.open(out, "w") as t:
        for _, _, stem, _, _ in todo:
            for i in range(25):
                s = "N_%s_t%d" % (stem, i)
                for ext, kind in (("jpg", "images"), ("txt", "labels")):
                    p = os.path.join(OUT_N, kind, s + "." + ext)
                    if os.path.exists(p):
                        t.add(p, arcname="tiles/normal/%s/%s.%s"
                              % (kind, s, ext))
                        cnt += 1
    print("TAR_OK %d 文件 %dMB" % (cnt, os.path.getsize(out) // 1048576))


if __name__ == "__main__":
    main()
