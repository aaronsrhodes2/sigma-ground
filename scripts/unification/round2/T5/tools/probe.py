import sys
from PIL import Image
import numpy as np
im = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
H, W, _ = im.shape
print(W, H)
# dark pixels (frame/ticks)
dark = (im.sum(axis=2) < 200)
# columns with many dark pixels in lower-left panel region (y 400..795, x 0..530)
sub = dark[400:795, 0:532]
colsum = sub.sum(axis=0); rowsum = sub.sum(axis=1)
print("dark cols:", [ (i, int(c)) for i, c in enumerate(colsum) if c > 100])
print("dark rows:", [ (i+400, int(c)) for i, c in enumerate(rowsum) if c > 150])
# colour census in panel
from collections import Counter
cnt = Counter(map(tuple, (im[400:795, 0:532].reshape(-1,3)//16*16)))
print(cnt.most_common(25))
