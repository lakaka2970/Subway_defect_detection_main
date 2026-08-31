# -*- coding: utf-8 -*-
"""DG-v2a 混合域 4 折 LOSO 训练（远端 5090）。

每折：train = 车间 tile + Normal tile + 其余折真实线 tile；
      val   = 车间 val tile + 训练折内 15% 真实线 val tile（混合 fitness）。
配置：init stage4_best.pt / epochs 20 / patience 6 / imgsz 1280 / batch 16 /
      AdamW lr0 1e-4 / cosine / warmup 2 / AMP / seed 42 / 全解冻。
不设类别权重（阶段1决策，见报告）；合成/整图分支未启用（消融留待后续）。
"""
import argparse
import json
import os
import shutil
import time

NAMES16 = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
           "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
           "BSBM", "INSD", "DRPS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="/root/autodl-tmp/subway/data/tiles")
    ap.add_argument("--weights",
                    default="/root/autodl-tmp/subway/weights/stage4_best.pt")
    ap.add_argument("--out", default="/root/autodl-tmp/subway/out/dgv2a")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--train-base", default="train_fold%d.txt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr0", type=float, default=1e-4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    # 让 ultralytics parse_model 识别自定义模块（yaml 中含 LSK/EMA/CoordAtt/SimAM）
    import ultralytics.nn.tasks as _UT
    from subway_yolo import (AuxClassifyHead, AuxHead, CoordAtt, DeformConv2d,
                             EMA, LSK, SimAM)
    for _n, _c in (("LSK", LSK), ("EMA", EMA), ("CoordAtt", CoordAtt),
                   ("SimAM", SimAM), ("AuxClassifyHead", AuxClassifyHead),
                   ("AuxHead", AuxHead), ("DeformConv2d", DeformConv2d)):
        setattr(_UT, _n, _c)

    # 预加载完整模型（含自定义模块），注入 trainer 以避免从 yaml 重建
    # weights 路径含 %d 时按折加载（第 2 轮从各折第 1 轮权重热启动）
    t_all = time.time()
    for f in [int(x) for x in args.folds.split(",")]:
        wpath = (args.weights % f) if "%d" in args.weights else args.weights
        base = YOLO(wpath)
        tr = os.path.join(args.tiles, "index", args.train_base % f)
        va = os.path.join(args.tiles, "index", "val_fold%d.txt" % f)
        yaml_p = os.path.join(args.out, "data_fold%d%s.yaml" % (f, args.tag))
        with open(yaml_p, "w", encoding="utf-8") as fp:
            fp.write("path: /\ntrain: %s\nval: %s\nnames:\n" % (tr, va))
            for i, n in enumerate(NAMES16):
                fp.write("  %d: %s\n" % (i, n))
        n_tr = sum(1 for _ in open(tr, encoding="utf-8"))
        n_va = sum(1 for _ in open(va, encoding="utf-8"))
        print("\n[FOLD %d] train %d / val %d  %s" % (f, n_tr, n_va,
                                                     time.strftime("%H:%M:%S")))
        t0 = time.time()
        trainer = DetectionTrainer(overrides=dict(
            model=args.weights, data=yaml_p, epochs=args.epochs, imgsz=1280,
            batch=args.batch, device=0, workers=8, amp=True, optimizer="AdamW",
            lr0=args.lr0, cos_lr=True, warmup_epochs=2.0, patience=args.patience,
            seed=42, project=args.out, name="fold%d%s" % (f, args.tag),
            exist_ok=True, cache=False, verbose=True, pretrained=False))
        trainer.model = base.model
        trainer.train()
        dt = (time.time() - t0) / 60
        src = os.path.join(args.out, "fold%d%s" % (f, args.tag),
                           "weights", "best.pt")
        dst = os.path.join(args.out, "fold%d%s_best.pt" % (f, args.tag))
        if os.path.exists(src):
            shutil.copy2(src, dst)
        print("[FOLD %d] 完成 %.1f 分钟 -> %s" % (f, dt, dst))
        with open(os.path.join(args.out, "fold%d%stime.json"
                               % (f, args.tag)), "w") as fp:
            json.dump({"minutes": round(dt, 1)}, fp)
    print("ALL_FOLDS_DONE %.1f h" % ((time.time() - t_all) / 3600))


if __name__ == "__main__":
    main()
