import sys, fitz
pdf, page, outprefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
doc = fitz.open(pdf)
p = doc[page - 1]
for i, info in enumerate(p.get_images(full=True)):
    xref = info[0]
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    fn = f"{outprefix}_img{i}.png"
    pix.save(fn)
    print(fn, pix.width, pix.height, p.get_image_bbox(info))
