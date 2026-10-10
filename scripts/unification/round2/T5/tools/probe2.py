import sys
from PIL import Image
import numpy as np
img = Image.open(sys.argv[1]).convert("RGB")
im = np.asarray(img).astype(int)
gray = (np.abs(im[:,:,0]-im[:,:,1]) < 20) & (np.abs(im[:,:,1]-im[:,:,2]) < 20) & (im.sum(axis=2) < 560)
sub = gray[400:795, 0:532]
colsum = sub.sum(axis=0); rowsum = sub.sum(axis=1)
print("gray cols:", [(i, int(c)) for i, c in enumerate(colsum) if c > 80])
print("gray rows:", [(i+400, int(c)) for i, c in enumerate(rowsum) if c > 80])
img.crop((60, 400, 532, 795)).resize((944, 790)).save(sys.argv[2])
