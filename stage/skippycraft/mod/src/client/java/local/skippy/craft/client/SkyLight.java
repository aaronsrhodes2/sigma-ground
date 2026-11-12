package local.skippy.craft.client;

import local.skippy.craft.SkyData;

/**
 * What a mirror shows of the sky: the geometry that turns a reflected ray into a lookup in sigma-ground's sky table
 * ({@link SkyData}). The engine's own sun angle says where the sun is (the sky renderer turns the sun by that angle about the
 * east-west axis: the direction is (-sin a, cos a, 0), noon at a = 0); the reflected ray's elevation and its angle round from the
 * sun's azimuth index the table. A ray that points down sees the ground. No physical quantity is computed here.
 */
public final class SkyLight {
	private static volatile float sunAngle = 0f;       // radians, as the engine's sky renderer holds it (the last frame's)

	private SkyLight() {}

	public static void sunAngle(float radians) {
		sunAngle = radians;
	}

	/** The sun's elevation in degrees (negative below the horizon). */
	public static double sunElevationDeg() {
		return Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, Math.cos(sunAngle)))));
	}

	/**
	 * The colour (r, g, b in 0..255) of the light arriving along the reflected ray {@code (rx, ry, rz)} (a unit vector pointing
	 * away from the surface), scaled by how much sky the surface sees (the game's own sky light, 0..1); null without a table.
	 */
	public static double[] reflected(double rx, double ry, double rz, double skyExposure) {
		SkyData d = SkyData.get();
		if (d == null) return null;
		double sunEl = sunElevationDeg();
		double[] c = new double[3];
		if (ry <= 0) {
			d.ground(sunEl, c);
		} else {
			double rh = Math.sqrt(rx * rx + rz * rz);
			double sunSide = -Math.signum(Math.sin(sunAngle));            // +1 when the sun is on the east side (+x), -1 on the west
			double cosPhi = rh < 1e-9 || sunSide == 0 ? 1 : Math.max(-1, Math.min(1, rx * sunSide / rh));
			d.sky(sunEl, Math.toDegrees(Math.asin(Math.min(1, ry))), Math.toDegrees(Math.acos(cosPhi)), c);
		}
		double k = 255 * Math.max(0, Math.min(1, skyExposure));
		return new double[] {c[0] * k, c[1] * k, c[2] * k};
	}
}
