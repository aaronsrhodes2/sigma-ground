"""T5 -- digitize published bound curves (read-only; images rendered from the cited PDFs).

Sources (each verified by web search 2026-10-10):
  Adelberger, Gundlach, Heckel, Hoedl, Schlamminger 2009, Prog. Part. Nucl. Phys. 62, 102-134
    Fig. 5 lower-left panel: 95% CL |alpha~| vs lambda for q~ = g~B  (curves EW99 = Smith et al. 2000; EW08 = Wagner PhD thesis)
    Fig. 9: 95% CL ISL |alpha| vs lambda, lambda <= 1 cm
    Fig. 10: 95% CL ISL |alpha| vs lambda, lambda >= 1 cm
Method: the excluded (yellow) region's lower boundary is located column by column; axes are calibrated on the
decade tick marks detected in the image (pixel positions recorded below). Reading uncertainty ~ +-0.05 dex.
Usage: python -I -X utf8 digitize_figures.py <lit_adelberger2009 dir> <out dir>
"""
import sys, json, os
import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]

def yellow_mask(im):
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    return (r > 200) & (g > 200) & (b < 175)

def lower_boundary(im, x0, x1, y0, y1):
    """For each column, the lower edge of the yellow region attached to the top of the frame."""
    ym = yellow_mask(im)
    global WHITE
    WHITE = (im[:, :, 0] > 232) & (im[:, :, 1] > 232) & (im[:, :, 2] > 232)
    cols = []
    for x in range(x0, x1 + 1):
        col = ym[y0:y1 + 1, x]
        wcol = WHITE[y0:y1 + 1, x]
        # walk down from the top of the frame through the excluded (yellow) region; the boundary is the first place
        # where >= 4 consecutive white pixels begin (rejects stray yellowish pixels in the allowed region)
        if not col[:6].any():
            cols.append((x, None)); continue
        yb = None
        for i in range(len(col) - 4):
            if wcol[i:i + 4].all():
                yb = i; break
        if yb is None:
            cols.append((x, None)); continue
        ys = np.nonzero(col[:yb])[0]
        if len(ys) == 0:
            cols.append((x, None)); continue
        # boundary line centre: mean row of the drawn-line pixels (neither yellow nor white) within +-6 px of the edge
        edge = int(ys.max())
        lo, hi = max(0, edge - 6), min(len(col), yb + 6)
        line = np.nonzero(~col[lo:hi] & ~wcol[lo:hi])[0]
        yc = (line.mean() + lo) if len(line) else 0.5 * (edge + yb)
        cols.append((x, float(yc) + y0))
    # running median over 7 columns removes spikes where other drawn lines cross the boundary
    ys = np.array([np.nan if y is None else y for _, y in cols], float)
    med = ys.copy()
    for i in range(len(ys)):
        w = ys[max(0, i - 3): i + 4]
        w = w[~np.isnan(w)]
        med[i] = np.median(w) if len(w) else np.nan
    return [(x, None if np.isnan(m) else float(m)) for (x, _), m in zip(cols, med)]

def blue_mask(im):
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    return (b > 150) & (r < 140) & (g < 140)

res = {}
# ---------------- Fig. 5, lower-left panel (q = B) ----------------
im5 = np.asarray(Image.open(os.path.join(src, "p11_img0.png")).convert("RGB")).astype(int)
XT = [120, 148.5, 177, 205, 233, 261.5, 290, 318, 346, 374.5, 403, 431, 459.5, 488, 516]   # decade ticks 1e-1 .. 1e13 m
YT = [410, 457, 504, 551, 597.5, 644.5, 691.5, 738.5]                                         # 1e-4 .. 1e-11
ax, bx = np.polyfit(XT, np.arange(-1, 14), 1)
ay, by = np.polyfit(YT, -np.arange(4, 12), 1)
fx5 = lambda x: ax * x + bx
fy5 = lambda y: ay * y + by
xr = (107, 515); yr = (411, 737)
b5 = lower_boundary(im5, *xr, *yr)
# also the lowest blue pixel cluster per column (EW08 is the lowest blue curve for lambda > ~1 m)
bm = blue_mask(im5)
blue_low = []
for x in range(*xr):
    ys = np.nonzero(bm[yr[0]:yr[1], x])[0]
    blue_low.append((x, int(ys.max()) + yr[0] if len(ys) else None))
res["fig5_B_panel"] = dict(
    source="Adelberger et al. 2009, PPNP 62, 102, Fig. 5 lower-left (q~ = g~B), 95% CL",
    calibration=dict(x_ticks_px=XT, x_ticks_log10_lambda=list(range(-1, 14)), y_ticks_px=YT,
                     y_ticks_log10_alpha=[-4, -5, -6, -7, -8, -9, -10, -11],
                     fit_x=[ax, bx], fit_y=[ay, by], plot_left_edge_log10_lambda=fx5(105.5)),
    yellow_lower_boundary=[(round(fx5(x), 4), round(fy5(y), 4)) for x, y in b5 if y is not None],
    lowest_blue=[(round(fx5(x), 4), round(fy5(y), 4)) for x, y in blue_low if y is not None],
)
# ---------------- Fig. 10 (ISL, lambda >= 1 cm) ----------------
im10 = np.asarray(Image.open(os.path.join(src, "p16_img1.png")).convert("RGB")).astype(int)
XT10 = [79.5, 107, 135, 163, 191, 219, 247, 275, 303, 331, 359, 387, 415, 442, 470.5, 498, 526, 554]  # 1e-2 .. 1e15
YT10 = [21, 63.5, 106, 149, 192, 235, 277.5, 320, 363, 406, 448]                                     # 1e-1 .. 1e-11
ax10, bx10 = np.polyfit(XT10, np.arange(-2, 16), 1)
ay10, by10 = np.polyfit(YT10, -np.arange(1, 12), 1)
b10 = lower_boundary(im10, 81, 553, 22, 447)
res["fig10_ISL"] = dict(
    source="Adelberger et al. 2009, PPNP 62, 102, Fig. 10 (ISL, lambda >= 1 cm), 95% CL",
    calibration=dict(x_ticks_px=XT10, y_ticks_px=YT10, fit_x=[ax10, bx10], fit_y=[ay10, by10]),
    yellow_lower_boundary=[(round(ax10 * x + bx10, 4), round(ay10 * y + by10, 4)) for x, y in b10 if y is not None],
)
# ---------------- Fig. 9 (ISL, lambda <= 1 cm) ----------------
im9 = np.asarray(Image.open(os.path.join(src, "p16_img0.png")).convert("RGB")).astype(int)
XT9 = [83.5, 202, 320.5, 439, 558]                       # 1e-6 .. 1e-2
YT9 = [22, 62, 101.5, 141, 180.5, 220, 259.5, 299, 338.5, 378, 417.5, 457, 496.5]   # 1e8 .. 1e-4
ax9, bx9 = np.polyfit(XT9, np.arange(-6, -1), 1)
ay9, by9 = np.polyfit(YT9, np.arange(8, -5, -1), 1)
b9 = lower_boundary(im9, 85, 557, 23, 495)
res["fig9_ISL"] = dict(
    source="Adelberger et al. 2009, PPNP 62, 102, Fig. 9 (ISL, lambda <= 1 cm), 95% CL",
    calibration=dict(x_ticks_px=XT9, y_ticks_px=YT9, fit_x=[ax9, bx9], fit_y=[ay9, by9]),
    yellow_lower_boundary=[(round(ax9 * x + bx9, 4), round(ay9 * y + by9, 4)) for x, y in b9 if y is not None],
)
json.dump(res, open(os.path.join(out, "digitized_bounds.json"), "w"), indent=0)

# overlays for visual verification
from PIL import ImageDraw
for name, fn, pts, fx, fy in (("fig5", "p11_img0.png", b5, None, None), ("fig10", "p16_img1.png", b10, None, None), ("fig9", "p16_img0.png", b9, None, None)):
    img = Image.open(os.path.join(src, fn)).convert("RGB")
    d = ImageDraw.Draw(img)
    for x, y in pts:
        if y is not None:
            d.point((x, y), fill=(255, 0, 0)); d.point((x, y - 1), fill=(255, 0, 0))
    img.save(os.path.join(out, f"overlay_{name}.png"))
print("fig5 x-fit px/dec", 1 / ax, "left edge log10 lambda", fx5(105.5), "| y px/dec", -1 / ay)
print("fig10 px/dec", 1 / ax10, -1 / ay10, "| fig9 px/dec", 1 / ax9, -1 / ay9)
