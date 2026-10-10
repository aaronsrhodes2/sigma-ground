import sys
from PIL import Image
import numpy as np
img = Image.open(sys.argv[1]).convert("RGB")
im = np.asarray(img).astype(int)
s = im.sum(axis=2)
dark = s < 400
# label text rows left of frame (x 0..100), y 400..795
rs = dark[400:795, 0:100].sum(axis=1)
runs=[]; inrun=False
for i,v in enumerate(rs):
    if v>0 and not inrun: start=i; inrun=True
    if v==0 and inrun: runs.append((start+400,i-1+400)); inrun=False
print("label text row-runs:", runs)
img.crop((0, 400, 140, 795)).resize((280, 790)).save(sys.argv[2])
