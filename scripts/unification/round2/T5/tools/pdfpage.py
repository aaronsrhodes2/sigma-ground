import sys, fitz
pdf, page, out, zoom = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
doc = fitz.open(pdf)
p = doc[page - 1]
clip = None
if len(sys.argv) > 5:
    x0, y0, x1, y1 = map(float, sys.argv[5].split(","))
    r = p.rect
    clip = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
pix.save(out)
print(pix.width, pix.height, p.rect)
