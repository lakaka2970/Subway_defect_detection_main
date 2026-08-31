# -*- coding: utf-8 -*-
"""把折索引引用的全部切片打包为单个 tar 供上传（远端布局与 txt 路径一致）。"""
import glob
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data", "tiles_index")
SRC = {"workshop": "tiles_workshop", "workshop_val": "tiles_workshop_val",
       "real": "tiles_real", "normal": "tiles_normal"}
RROOT = "/root/autodl-tmp/subway/data/tiles"


def main():
    stems = {k: set() for k in SRC}
    for f in glob.glob(os.path.join(IDX, "*_fold*.txt")):
        for ln in open(f, encoding="utf-8"):
            p = ln.strip()
            if not p.startswith(RROOT):
                continue
            rel = p[len(RROOT) + 1:]            # workshop/images/X.jpg
            sub, _, fn = rel.split("/", 2)
            if sub in stems:
                stems[sub].add(os.path.splitext(fn)[0])
    out = os.path.join(ROOT, "data", "_upload", "tiles_phase1.tar")
    n = 0
    with tarfile.open(out, "w") as t:
        for sub, d in SRC.items():
            for s in sorted(stems[sub]):
                for ext, kind in (("jpg", "images"), ("txt", "labels")):
                    p = os.path.join(ROOT, "data", d, kind, s + "." + ext)
                    if os.path.exists(p):
                        t.add(p, arcname="tiles/%s/%s/%s.%s"
                              % (sub, kind, s, ext))
                        n += 1
        for f in sorted(glob.glob(os.path.join(IDX, "*.txt"))):
            t.add(f, arcname="tiles/index/" + os.path.basename(f))
            n += 1
    print("TAR2_OK files=%d size=%dMB" % (n, os.path.getsize(out) // 1048576))


if __name__ == "__main__":
    main()
