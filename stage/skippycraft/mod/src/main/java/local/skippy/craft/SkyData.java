package local.skippy.craft;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * sigma-ground's clear-sky table (field.interface.sky, written by mod/gen_sky.py): the display colour of the sky by the sun's
 * elevation, the view's elevation and its azimuth from the sun, and of the ground by the sun's elevation. This class only
 * interpolates it; it computes no physical quantity.
 */
public final class SkyData {
	private static SkyData instance;
	private static boolean tried;

	private final double[] sun, view, az;
	private final double[][][][] sky;      // [sun][view][azimuth][r g b], sRGB-encoded 0..1 against the white card
	private final double[][] ground;       // [sun][r g b]
	private final String waterMaterial;    // the material the water table is for, or null
	private final double[] wSun, wView, wAz;                   // the water table's own (finer) axes
	private final double[][][][] water;    // [sun][view][azimuth][r g b alpha]: colour sRGB-encoded 0..1, alpha 0..1

	/** The table, or null when the resource is missing (the renderer then falls back to a white environment). */
	public static synchronized SkyData get() {
		if (!tried) {
			tried = true;
			try (var in = SkyData.class.getResourceAsStream("/skippycraft_sky.json")) {
				if (in != null) instance = new SkyData(JsonParser.parseReader(new InputStreamReader(in, StandardCharsets.UTF_8)).getAsJsonObject());
			} catch (Exception e) {
				SkippyCraftMod.LOGGER.warn("skippycraft: the sky table could not be read", e);
			}
		}
		return instance;
	}

	private SkyData(JsonObject o) {
		sun = axis(o.getAsJsonArray("sun_elevation_deg"));
		view = axis(o.getAsJsonArray("view_elevation_deg"));
		az = axis(o.getAsJsonArray("relative_azimuth_deg"));
		JsonArray s = o.getAsJsonArray("sky");
		sky = new double[sun.length][view.length][az.length][];
		for (int i = 0; i < sun.length; i++) for (int j = 0; j < view.length; j++) for (int k = 0; k < az.length; k++) {
			sky[i][j][k] = axis(s.get(i).getAsJsonArray().get(j).getAsJsonArray().get(k).getAsJsonArray());
		}
		JsonArray g = o.getAsJsonArray("ground");
		ground = new double[sun.length][];
		for (int i = 0; i < sun.length; i++) ground[i] = axis(g.get(i).getAsJsonArray());
		JsonObject w = o.has("water") ? o.getAsJsonObject("water") : null;
		if (w == null) {
			waterMaterial = null;
			wSun = wView = wAz = null;
			water = null;
		} else {
			waterMaterial = w.get("material").getAsString();
			wSun = axis(w.getAsJsonArray("sun_elevation_deg"));
			wView = axis(w.getAsJsonArray("view_elevation_deg"));
			wAz = axis(w.getAsJsonArray("relative_azimuth_deg"));
			JsonArray c = w.getAsJsonArray("rgb_display"), a = w.getAsJsonArray("alpha");
			water = new double[wSun.length][wView.length][wAz.length][];
			for (int i = 0; i < wSun.length; i++) for (int j = 0; j < wView.length; j++) for (int k = 0; k < wAz.length; k++) {
				double[] rgb = axis(c.get(i).getAsJsonArray().get(j).getAsJsonArray().get(k).getAsJsonArray());
				water[i][j][k] = new double[] {rgb[0], rgb[1], rgb[2], a.get(i).getAsJsonArray().get(j).getAsJsonArray().get(k).getAsDouble()};
			}
		}
	}

	/** The material the water table is for (the one with a wind-roughened surface that glints), or null without a water table. */
	public String waterMaterial() {
		return waterMaterial;
	}

	/**
	 * A horizontal sheet of water seen by its reflected ray: the face's display colour (r, g, b in 0..1) and its alpha, the sky's
	 * reflection plus the sun's glint on the waves, to composite over what is behind it. {@code out} has four entries.
	 */
	public void water(double sunDeg, double viewDeg, double azDeg, double[] out) {
		int[] si = new int[2], vi = new int[2], ai = new int[2];
		double[] sf = new double[1], vf = new double[1], af = new double[1];
		locate(wSun, sunDeg, si, sf);
		locate(wView, viewDeg, vi, vf);
		locate(wAz, azDeg, ai, af);
		out[0] = out[1] = out[2] = out[3] = 0;
		for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
			double w = (a == 0 ? 1 - sf[0] : sf[0]) * (b == 0 ? 1 - vf[0] : vf[0]) * (c == 0 ? 1 - af[0] : af[0]);
			if (w == 0) continue;
			double[] v = water[si[a]][vi[b]][ai[c]];
			for (int k = 0; k < 4; k++) out[k] += w * v[k];
		}
	}

	/** The sky's display colour (r, g, b in 0..1) looking up at this elevation and this angle round from the sun. */
	public void sky(double sunDeg, double viewDeg, double azDeg, double[] out) {
		int[] si = new int[2], vi = new int[2], ai = new int[2];
		double[] sf = new double[1], vf = new double[1], af = new double[1];
		locate(sun, sunDeg, si, sf);
		locate(view, viewDeg, vi, vf);
		locate(az, azDeg, ai, af);
		out[0] = out[1] = out[2] = 0;
		for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
			double w = (a == 0 ? 1 - sf[0] : sf[0]) * (b == 0 ? 1 - vf[0] : vf[0]) * (c == 0 ? 1 - af[0] : af[0]);
			if (w == 0) continue;
			double[] v = sky[si[a]][vi[b]][ai[c]];
			out[0] += w * v[0];
			out[1] += w * v[1];
			out[2] += w * v[2];
		}
	}

	/** The ground's display colour (r, g, b in 0..1) under a sun at this elevation. */
	public void ground(double sunDeg, double[] out) {
		int[] si = new int[2];
		double[] sf = new double[1];
		locate(sun, sunDeg, si, sf);
		for (int c = 0; c < 3; c++) out[c] = ground[si[0]][c] * (1 - sf[0]) + ground[si[1]][c] * sf[0];
	}

	/** The two table indices around x on a sorted axis and the weight of the upper one; x beyond the ends is clamped to them. */
	private static void locate(double[] axis, double x, int[] idx, double[] frac) {
		if (x <= axis[0]) {
			idx[0] = idx[1] = 0;
			frac[0] = 0;
			return;
		}
		int n = axis.length;
		if (x >= axis[n - 1]) {
			idx[0] = idx[1] = n - 1;
			frac[0] = 0;
			return;
		}
		int i = 0;
		while (axis[i + 1] < x) i++;
		idx[0] = i;
		idx[1] = i + 1;
		frac[0] = (x - axis[i]) / (axis[i + 1] - axis[i]);
	}

	private static double[] axis(JsonArray a) {
		double[] r = new double[a.size()];
		for (int i = 0; i < r.length; i++) r[i] = a.get(i).getAsDouble();
		return r;
	}
}
