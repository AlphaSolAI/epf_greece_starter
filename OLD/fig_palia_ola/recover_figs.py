"""One-shot: recover figures from 106.pdf, map to thesis filenames, ready for compile."""
import subprocess, re
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
PDF  = Path("/sessions/awesome-bold-rubin/mnt/uploads/106.pdf")
EX   = Path("/tmp/ex")
OUT  = ROOT / "thesis_output"

# 1) real figure images from the extracted set (page order via filename), skip greyscale smasks/tiny
imgs = []
for p in sorted(EX.glob("*.png")):
    try:
        im = Image.open(p)
        if im.width > 400 and im.mode in ("RGB", "RGBA", "P"):
            # skip near-grayscale smasks that slipped through
            imgs.append(p)
    except Exception:
        pass
print("real images:", len(imgs))

# 2) tex figure refs in document order (exclude ui screenshots + my regenerated error-heatmaps)
order = ["introduction","literature","data","methods","experiments","forecast_analysis",
         "feature_importance","statistical_tests","conclusions","appendix_fi"]
refs = []
for f in order:
    t = ROOT / "thesis" / "content" / f"{f}.tex"
    if not t.exists():
        continue
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t.read_text(encoding="utf-8")):
        r = m.group(1).strip()
        if "ui_screenshots" in r or "heatmap_error" in r:
            continue
        if not r.lower().endswith((".png",".pdf",".jpg",".jpeg")):
            r = r + ".png"
        refs.append(r)
print("tex refs:", len(refs))

# 3) map in order and save into thesis_output (real figures overwrite placeholders)
n = min(len(refs), len(imgs))
for i in range(n):
    dst = OUT / refs[i]
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(imgs[i]).convert("RGB").save(dst)
print(f"mapped {n} figures into thesis_output/")
if len(refs) > n:
    print("UNMAPPED refs (no image):", refs[n:])
