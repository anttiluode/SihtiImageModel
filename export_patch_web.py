"""Export the patch memories (P0 and P3 arms) and a few held-out BSDS photos for the web page."""
import json
import os
import pickle
import shutil

from PIL import Image

from export_web import q16

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = {"42049": "val", "8023": "val", "108082": "val", "196073": "val"}


def main():
    web = os.path.join(HERE, "web")
    for ds in ["mnist", "fashion", "bsds"]:
        pm = pickle.load(open(os.path.join(HERE, "results", "cache", f"patch_{ds}.pkl"), "rb"))
        out = {"K": pm.K, "bands": {}}
        for b, a in pm.arms.items():
            out["bands"][b] = {"C": q16(a["C"]), "W": q16(a["W"]), "W0": q16(a["W0"])}
        p = os.path.join(web, f"patch_{ds}.json")
        json.dump(out, open(p, "w"))
        print(p, f"{os.path.getsize(p) / 1e6:.1f} MB")
        r = os.path.join(HERE, "results", f"patch_results_{ds}.json")
        shutil.copy(r, os.path.join(web, f"patch_results_{ds}.json"))
    root = os.path.join(HERE, "data", "bsds", "BSDS500", "data", "images")
    for n, split in PHOTOS.items():
        im = Image.open(os.path.join(root, split, n + ".jpg")).convert("RGB")
        im = im.resize((im.width // 2, im.height // 2), Image.BICUBIC)
        im.save(os.path.join(web, f"photo_{n}.png"))
    print("photos", list(PHOTOS))


if __name__ == "__main__":
    main()
