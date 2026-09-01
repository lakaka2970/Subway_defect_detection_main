# -*- coding: utf-8 -*-
"""交付候选模型训练：全量数据（无留出折），用于 10/15 部署。

配方 = 第 2 轮 r2 配方（硬负样本过采样 + 全 931 Normal + 全部 534 真实图）；
init stage4_best.pt；12 epochs；lr0 1e-4；batch 12；seed 42。
指标口径声明：本模型训练见过全部基准图，其自测数字仅作收敛 sanity，
交付能力以四折 LOSO 模型为代表。
"""
import json
import os
import shutil
import time

NAMES16 = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
           "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
           "BSBM", "INSD", "DRPS"]


def main():
    root = "/root/autodl-tmp/subway"
    tiles = os.path.join(root, "data/tiles")
    out = os.path.join(root, "out/deploy")
    os.makedirs(out, exist_ok=True)
    yaml_p = os.path.join(out, "data_deploy.yaml")
    with open(yaml_p, "w", encoding="utf-8") as fp:
        fp.write("path: /\ntrain: %s/index/deploy_train.txt\n"
                 "val: %s/index/deploy_val.txt\nnames:\n" % (tiles, tiles))
        for i, n in enumerate(NAMES16):
            fp.write("  %d: %s\n" % (i, n))
    n_tr = sum(1 for _ in open(os.path.join(tiles, "index",
                                            "deploy_train.txt"),
                               encoding="utf-8"))
    print("[DEPLOY] train %d" % n_tr)

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    import ultralytics.nn.tasks as _UT
    from subway_yolo import (AuxClassifyHead, AuxHead, CoordAtt,
                             DeformConv2d, EMA, LSK, SimAM)
    for _n, _c in (("LSK", LSK), ("EMA", EMA), ("CoordAtt", CoordAtt),
                   ("SimAM", SimAM), ("AuxClassifyHead", AuxClassifyHead),
                   ("AuxHead", AuxHead), ("DeformConv2d", DeformConv2d)):
        setattr(_UT, _n, _c)
    w = os.path.join(root, "weights/stage4_best.pt")
    base = YOLO(w)
    t0 = time.time()
    trainer = DetectionTrainer(overrides=dict(
        model=w, data=yaml_p, epochs=12, imgsz=1280, batch=12, device=0,
        workers=8, amp=True, optimizer="AdamW", lr0=1e-4, cos_lr=True,
        warmup_epochs=2.0, patience=6, seed=42, project=out, name="deploy",
        exist_ok=True, cache=False, verbose=True, pretrained=False))
    trainer.model = base.model
    trainer.train()
    src = os.path.join(out, "deploy", "weights", "best.pt")
    dst = os.path.join(out, "deploy_best.pt")
    if os.path.exists(src):
        shutil.copy2(src, dst)
    print("[DEPLOY] 完成 %.1f 分钟 -> %s"
          % ((time.time() - t0) / 60, dst))
    with open(os.path.join(out, "deploy_time.json"), "w") as f:
        json.dump({"minutes": round((time.time() - t0) / 60, 1)}, f)
    print("DEPLOY_DONE")


if __name__ == "__main__":
    main()
