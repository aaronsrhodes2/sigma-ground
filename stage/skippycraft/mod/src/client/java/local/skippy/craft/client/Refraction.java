package local.skippy.craft.client;

import local.skippy.craft.MatterBlockEntity;
import local.skippy.craft.MatterData;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.Level;
import net.minecraft.world.phys.Vec3;

import java.util.HashMap;
import java.util.Map;

/**
 * Where an opaque cube is SEEN when clear matter lies between it and the camera: the apparent position of a point under a
 * refracting surface (the gold at the bottom of the tank looks shallower and shifted, the way it does through real water).
 *
 * It computes no physics. The refraction angle at the interface is read from the table sigma-ground's optics wrote into the
 * medium's record ({@link MatterData.Tables#cosRefractedIn}); everything else here is geometry: a voxel walk from the camera to
 * find the surface the ray enters through and the medium it is in, and a search for the entry point whose refracted ray lands on
 * the vertex. The vertex is then drawn on the camera's line of sight through that entry point, where the refracted ray's
 * extension meets the vertical through the vertex (the apparent position).
 *
 * What it assumes, named in KNOWN_GAPS.md: one planar entry surface (the first face of clear matter along the ray), the
 * medium of the last clear block before the vertex, and faces between the two parallel (layers in between shift the ray
 * sideways and are ignored); light from the vertex that leaves the body and re-enters is not followed; only the cubes of
 * matter blocks move, not the vanilla blocks behind a tank, which a screen-space pass (not available in this renderer) would need.
 */
final class Refraction {
	private static final double[] ZERO = new double[3];
	private static final double MAX_REACH = 160;       // blocks; a vertex farther than this is not refracted
	private static final int BISECTIONS = 24;

	private static long cacheTick = Long.MIN_VALUE;
	private static Vec3 cacheCamera = Vec3.ZERO;
	private static final Map<Long, double[]> CACHE = new HashMap<>();

	/** On unless a test (or the Captain) sets -Dskippycraft.refraction=false. */
	static boolean enabled() {
		return !"false".equals(System.getProperty("skippycraft.refraction"));
	}

	/** The shift of each corner of the block at {@code pos} (bit 0 x, bit 1 y, bit 2 z), or null when none moves. */
	static double[][] corners(Level level, Vec3 camera, BlockPos pos) {
		long tick = level.getGameTime();
		if (tick != cacheTick || !camera.equals(cacheCamera)) {          // the world or the camera moved: start again
			CACHE.clear();
			cacheTick = tick;
			cacheCamera = camera;
		}
		double[][] out = new double[8][];
		boolean any = false;
		for (int i = 0; i < 8; i++) {
			int vx = pos.getX() + (i & 1), vy = pos.getY() + ((i >> 1) & 1), vz = pos.getZ() + ((i >> 2) & 1);
			long key = ((long) (vx & 0x1FFFFF) << 42) | ((long) (vy & 0x1FFFFF) << 21) | (vz & 0x1FFFFF);
			double[] d = CACHE.get(key);
			if (d == null) {
				d = shift(level, camera, vx, vy, vz);
				CACHE.put(key, d);
			}
			out[i] = d;
			any |= d != ZERO;
		}
		return any ? out : null;
	}

	private static MatterData clearAt(Level level, BlockPos pos) {
		return level.getBlockEntity(pos) instanceof MatterBlockEntity m && m.data().clear() ? m.data() : null;
	}

	private static double[] shift(Level level, Vec3 cam, double vx, double vy, double vz) {
		double dx = vx - cam.x, dy = vy - cam.y, dz = vz - cam.z;
		double len = Math.sqrt(dx * dx + dy * dy + dz * dz);
		if (len < 1.5 || len > MAX_REACH) return ZERO;
		double[] u = {dx / len, dy / len, dz / len};
		double reach = len - 0.6;                                          // stop just short of the vertex's own cell

		// the voxel walk (Amanatides and Woo) from the camera to the vertex
		int[] cell = {(int) Math.floor(cam.x), (int) Math.floor(cam.y), (int) Math.floor(cam.z)};
		double[] origin = {cam.x, cam.y, cam.z};
		int[] step = new int[3];
		double[] tMax = new double[3], tDelta = new double[3];
		for (int a = 0; a < 3; a++) {
			step[a] = u[a] > 0 ? 1 : -1;
			if (u[a] == 0) {
				tMax[a] = tDelta[a] = Double.POSITIVE_INFINITY;
			} else {
				tMax[a] = ((u[a] > 0 ? cell[a] + 1 : cell[a]) - origin[a]) / u[a];
				tDelta[a] = Math.abs(1.0 / u[a]);
			}
		}
		boolean inside = false;
		double entryT = 0, lastT = 0;
		int entryAxis = -1, lastAxis = -1;
		MatterData medium = null;
		BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
		while (true) {
			MatterData d = clearAt(level, p.set(cell[0], cell[1], cell[2]));
			if (d != null) {
				if (!inside) {
					inside = true;
					entryT = lastT;
					entryAxis = lastAxis;
				}
				medium = d;
			} else {
				inside = false;
			}
			int axis = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : (tMax[1] < tMax[2] ? 1 : 2);
			if (tMax[axis] > reach) break;
			lastT = tMax[axis];
			lastAxis = axis;
			cell[axis] += step[axis];
			tMax[axis] += tDelta[axis];
		}
		if (!inside || medium == null || entryAxis < 0) return ZERO;   // not in a medium, or the camera is in it

		// the entry surface: its outward normal faces the camera
		double[] n = new double[3];
		n[entryAxis] = -step[entryAxis];
		double[] sIn = {cam.x + u[0] * entryT, cam.y + u[1] * entryT, cam.z + u[2] * entryT};
		double h = -entryT * dot(u, n);                                    // the camera's height above the surface
		double depth = (sIn[0] - vx) * n[0] + (sIn[1] - vy) * n[1] + (sIn[2] - vz) * n[2];   // the vertex's depth below it
		if (h <= 1e-6 || depth <= 1e-6) return ZERO;
		double[] p0 = {cam.x - n[0] * h, cam.y - n[1] * h, cam.z - n[2] * h};     // the camera's foot on the surface
		double[] w = {vx - p0[0], vy - p0[1], vz - p0[2]};
		double wn = dot(w, n);
		for (int a = 0; a < 3; a++) w[a] -= n[a] * wn;
		double x = Math.sqrt(dot(w, w));                                   // the vertex's lateral offset from that foot
		if (x < 1e-6) return ZERO;                                         // straight down the normal: nothing shifts
		double[] e = {w[0] / x, w[1] / x, w[2] / x};

		// the entry point whose refracted ray lands on the vertex: u + depth * tan(theta_t) = x, theta_t from sigma-ground's table
		MatterData.Tables tab = medium.tables;
		double lo = 0, hi = x;
		for (int i = 0; i < BISECTIONS; i++) {
			double mid = (lo + hi) / 2;
			double cosI = h / Math.sqrt(h * h + mid * mid);
			double cosT = Math.max(tab.cosRefractedIn(cosI), 1e-3);
			double land = mid + depth * Math.sqrt(Math.max(0, 1 - cosT * cosT)) / cosT;
			if (land > x) hi = mid; else lo = mid;
		}
		double lateral = (lo + hi) / 2;
		if (lateral < 1e-9) return ZERO;
		double[] s = {p0[0] + e[0] * lateral, p0[1] + e[1] * lateral, p0[2] + e[2] * lateral};
		double[] cs = {s[0] - cam.x, s[1] - cam.y, s[2] - cam.z};
		double csLen = Math.sqrt(dot(cs, cs));
		double along = csLen * (x / lateral);                              // where the line of sight meets the vertical through the vertex
		return new double[] {cam.x + cs[0] / csLen * along - vx, cam.y + cs[1] / csLen * along - vy, cam.z + cs[2] / csLen * along - vz};
	}

	private static double dot(double[] a, double[] b) {
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	}
}
