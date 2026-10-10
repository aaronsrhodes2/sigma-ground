import sys
from PIL import Image
import numpy as np
im = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
H, W, _ = im.shape
s = im.sum(axis=2)
dark = (s < 330)
# frame: longest dark rows/cols
rows = dark.sum(axis=1); cols = dark.sum(axis=0)
fr = [i for i in range(H) if rows[i] > 0.6 * W]; fc = [j for j in range(W) if cols[j] > 0.6 * H]
print("W,H", W, H, "frame rows", fr, "frame cols", fc)
y0, y1 = min(fr), max(fr); x0, x1 = min(fc), max(fc)
# x ticks: just above bottom frame
band = dark[y1-7:y1-1, x0+2:x1-1].sum(axis=0)
print("xticks", [j + x0 + 2 for j, v in enumerate(band) if v >= 4])
band = dark[y0+1:y1-1, x0+2:x0+8].sum(axis=1)
print("yticks", [i + y0 + 1 for i, v in enumerate(band) if v >= 4])
