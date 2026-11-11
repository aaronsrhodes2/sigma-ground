package local.skippy.craft;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * One block's matter record (stage/skippycraft/matter.py is the writer): what the block IS and how it is drawn.
 * The JSON text is the record; this reads the fields the game needs from it. Time lives only where it is needed:
 * {@code t} is NaN for a block with no clock.
 */
public final class MatterData {
	public static final MatterData EMPTY = new MatterData("");
	private static final java.util.Map<String, MatterData> CACHE = new java.util.concurrent.ConcurrentHashMap<>();

	/** The parsed record for this JSON, shared: a whole body of water is thousands of blocks with one record. */
	public static MatterData of(String json) {
		if (json.isEmpty()) return EMPTY;
		if (CACHE.size() > 4096) CACHE.clear();
		return CACHE.computeIfAbsent(json, MatterData::new);
	}

	public final String json;
	public final String material;
	public final String phase;
	public final double massKg;
	public final double pitchM;
	public final double temperatureK;
	public final double t;
	public final double[] v;
	public final int[] rgba;
	public final int[] glowRgb;
	public final int light;
	public final double emission;
	public final double opacityPhysical;
	public final double drawnAtM;
	public final boolean metal;
	public final double smoothness;
	public final double melt;
	public final double n;                 // refractive index at 550 nm; NaN for a material light does not pass through
	public final double[] absorption;      // bulk absorption per metre, R G B; null for an opaque material
	public final Tables tables;            // sigma-ground's optics, sampled: the only physics the renderer draws with
	public final List<String> gaps;

	public MatterData(String json) {
		this.json = json;
		JsonObject o = json.isEmpty() ? new JsonObject() : JsonParser.parseString(json).getAsJsonObject();
		material = str(o, "material", "?");
		phase = str(o, "phase", "");
		massKg = num(o, "mass_kg", 0);
		pitchM = num(o, "pitch_m", 0);
		temperatureK = num(o, "temperature_k", 293.15);
		t = num(o, "t", Double.NaN);
		v = vec(o, "v", 3, 0);
		rgba = ints(o, "rgba", new int[] {125, 125, 125, 255});
		glowRgb = ints(o, "glow_rgb", new int[] {0, 0, 0});
		light = (int) num(o, "light", 0);
		emission = num(o, "emission", 0);
		opacityPhysical = num(o, "opacity_physical", 1);
		drawnAtM = num(o, "drawn_at_m", Double.NaN);
		metal = o.has("metal") && !o.get("metal").isJsonNull() && o.get("metal").getAsBoolean();
		smoothness = num(o, "smoothness", Double.NaN);
		melt = num(o, "melt", 0);
		n = num(o, "n", Double.NaN);
		absorption = o.has("absorption_per_m") && o.get("absorption_per_m").isJsonArray() ? vec(o, "absorption_per_m", 3, 0) : null;
		tables = o.has("tables") && o.get("tables").isJsonObject() ? new Tables(o.getAsJsonObject("tables")) : null;
		gaps = new ArrayList<>();
		if (o.has("gaps") && o.get("gaps").isJsonArray()) {
			for (JsonElement g : o.getAsJsonArray("gaps")) gaps.add(g.getAsString());
		}
	}

	/**
	 * sigma-ground's optics for a clear material, sampled by the writer (stage/skippycraft/matter.py): for light going air into
	 * the material ("in") and back out ("out"), the Fresnel reflectance and the cosine of the refraction angle against the
	 * cosine of incidence (uniform on [0, 1]); and the transmittance of 0, 1, 2, ... cells of the material (R, G, B). This class
	 * only interpolates them. It computes no physical quantity.
	 */
	public static final class Tables {
		private final double[] inR, inCosT, outR, outCosT;
		private final double[][] trans;

		Tables(JsonObject o) {
			inR = arr(o.getAsJsonObject("in"), "reflectance");
			inCosT = arr(o.getAsJsonObject("in"), "cos_t");
			outR = arr(o.getAsJsonObject("out"), "reflectance");
			outCosT = arr(o.getAsJsonObject("out"), "cos_t");
			JsonArray rgb = o.getAsJsonObject("transmittance").getAsJsonArray("rgb");
			trans = new double[rgb.size()][];
			for (int c = 0; c < trans.length; c++) trans[c] = toArray(rgb.get(c).getAsJsonArray());
		}

		/** Reflected fraction for light entering the material at this cosine of incidence. */
		public double reflectanceIn(double cosI) { return at(inR, cosI); }

		/** Cosine of the refraction angle for light entering the material at this cosine of incidence. */
		public double cosRefractedIn(double cosI) { return at(inCosT, cosI); }

		/** Reflected fraction for light inside the material meeting its surface at this cosine of incidence. */
		public double reflectanceOut(double cosI) { return at(outR, cosI); }

		/** Mean transmittance (over R, G, B) of this many cells of the material, interpolating between whole cells. */
		public double transmittance(double cells) {
			double sum = 0;
			for (double[] ch : trans) sum += at(ch, cells / (ch.length - 1));
			return sum / trans.length;
		}

		/** Transmittance of this many cells of the material in each of R, G, B (a multiply per channel), interpolating between whole cells. */
		public double[] transmittanceRgb(double cells) {
			double[] out = new double[trans.length];
			for (int c = 0; c < trans.length; c++) out[c] = at(trans[c], cells / (trans[c].length - 1));
			return out;
		}

		/** Linear interpolation in a table sampled uniformly over [0, 1]; x is clamped to it. */
		private static double at(double[] ys, double x) {
			double f = Math.max(0, Math.min(1, x)) * (ys.length - 1);
			int i = Math.min((int) f, ys.length - 2);
			return ys[i] + (ys[i + 1] - ys[i]) * (f - i);
		}

		private static double[] arr(JsonObject o, String k) {
			return toArray(o.getAsJsonArray(k));
		}

		private static double[] toArray(JsonArray a) {
			double[] r = new double[a.size()];
			for (int i = 0; i < r.length; i++) r[i] = a.get(i).getAsDouble();
			return r;
		}
	}

	public boolean isEmpty() {
		return json.isEmpty();
	}

	/** A clear dielectric: drawn from sigma-ground's optics tables, not from one opacity. */
	public boolean clear() {
		return tables != null;
	}

	public boolean opaque() {
		return rgba[3] >= 255;
	}

	/** A one-line account of the block for the action bar. */
	public String summary() {
		StringBuilder s = new StringBuilder(material).append("  ").append(fmt(massKg)).append(" kg  ")
			.append(Math.round(temperatureK)).append(" K");
		if (!phase.isEmpty()) s.append("  ").append(phase);
		if (!Double.isNaN(t)) {
			double sp = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
			s.append("  t=").append(fmt(t)).append(" s  |v|=").append(fmt(sp)).append(" m/s");
		}
		if (!gaps.isEmpty()) s.append("  [").append(gaps.size()).append(" gaps]");
		return s.toString();
	}

	private static String fmt(double x) {
		if (x != 0 && (Math.abs(x) < 0.01 || Math.abs(x) >= 1e5)) return String.format("%.3g", x);
		return String.format("%.3f", x).replaceAll("\\.?0+$", "");
	}

	private static String str(JsonObject o, String k, String d) {
		return o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsString() : d;
	}

	private static double num(JsonObject o, String k, double d) {
		return o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsDouble() : d;
	}

	private static double[] vec(JsonObject o, String k, int n, double d) {
		double[] r = new double[n];
		Arrays.fill(r, d);
		if (o.has(k) && o.get(k).isJsonArray()) {
			JsonArray a = o.getAsJsonArray(k);
			for (int i = 0; i < n && i < a.size(); i++) r[i] = a.get(i).getAsDouble();
		}
		return r;
	}

	private static int[] ints(JsonObject o, String k, int[] d) {
		if (!o.has(k) || !o.get(k).isJsonArray()) return d;
		JsonArray a = o.getAsJsonArray(k);
		int[] r = new int[a.size()];
		for (int i = 0; i < r.length; i++) r[i] = a.get(i).getAsInt();
		return r.length >= d.length ? r : d;
	}
}
