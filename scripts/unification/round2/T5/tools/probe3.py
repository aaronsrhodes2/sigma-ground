import sys
from PIL import Image
import numpy as np
im = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
s = im.sum(axis=2)
grayish = (np.abs(im[:,:,0]-im[:,:,2]) < 40) & (s < 600)
# x ticks: columns between x=106..530 with dark pixels in rows 726..736 (just above bottom frame)
band = grayish[724:737, 100:540]
cs = band.sum(axis=0)
print("x-tick candidate cols:", [(i+100, int(c)) for i, c in enumerate(cs) if c >= 5])
# y ticks: rows with dark pixels in cols 107..118 (right of left frame)
band = grayish[400:740, 107:119]
rs = band.sum(axis=1)
print("y-tick candidate rows:", [(i+400, int(c)) for i, c in enumerate(rs) if c >= 5])
# top frame: rows above 420 where many grayish in x 106..530
for y in range(395, 425):
    print(y, int(grayish[y, 106:530].sum()), end="; ")
print()
# right frame col
for x in range(520, 535):
    print(x, int(grayish[400:740, x].sum()), end="; ")
