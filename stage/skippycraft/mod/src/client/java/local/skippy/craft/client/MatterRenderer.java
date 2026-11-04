package local.skippy.craft.client;

import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.VertexConsumer;
import local.skippy.craft.MatterBlock;
import local.skippy.craft.MatterBlockEntity;
import local.skippy.craft.MatterData;
import local.skippy.craft.SkippyCraftMod;
import net.minecraft.client.renderer.SubmitNodeCollector;
import net.minecraft.client.renderer.blockentity.BlockEntityRenderer;
import net.minecraft.client.renderer.blockentity.state.BlockEntityRenderState;
import net.minecraft.client.renderer.feature.ModelFeatureRenderer;
import net.minecraft.client.renderer.rendertype.RenderType;
import net.minecraft.client.renderer.rendertype.RenderTypes;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.texture.OverlayTexture;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.phys.Vec3;

/**
 * Draws a matter block from its record, with nothing tuned for the screen.
 *
 * An opaque material is a cube in its record's colour, brightened toward its glow colour by its emission, lit as bright as
 * it glows.
 *
 * A clear dielectric (water, glass, ice, ...) is drawn the way light meets it: each face reflects the Fresnel fraction F of
 * the light around it (exact unpolarised Fresnel for the record's index n, at the angle the camera actually sees the face),
 * and what is not reflected goes on through the body and is attenuated by Beer-Lambert over the real path length: the
 * blocks of the same material behind the face (counted here), at the refracted angle. Faces between two blocks of the same
 * material are not drawn: there is no interface there. The environment a face reflects is white at the local light level
 * (no environment map), a back face's reflectance uses the parallel-face angle, and the attenuation is one number (the mean
 * of R, G, B) because alpha blending cannot multiply per channel. Each of these is named in KNOWN_GAPS.md; refraction (the
 * bending that makes clear things look clear) needs a screen-space pass and is the next gap to close.
 */
public final class MatterRenderer implements BlockEntityRenderer<MatterBlockEntity, MatterRenderer.State> {
	private static final Identifier WHITE = SkippyCraftMod.id("textures/misc/white.png");
	private static final int FULL_BRIGHT = 0xF000F0;
	private static final int MAX_BODY = 256;

	public static final class State extends BlockEntityRenderState {
		MatterData data = MatterData.EMPTY;
		int hidden;                         // bit i set: the face toward Direction.values()[i] is covered
		final int[] depth = new int[6];     // blocks of the same material behind each face (a clear material)
		double cx, cy, cz;                  // the camera, relative to this block's corner
	}

	@Override
	public State createRenderState() {
		return new State();
	}

	@Override
	public void extractRenderState(MatterBlockEntity be, State s, float partialTick, Vec3 camera,
								   ModelFeatureRenderer.CrumblingOverlay crumbling) {
		BlockEntityRenderState.extractBase(be, s, crumbling);
		s.data = be.data();
		s.hidden = 0;
		BlockPos pos = be.getBlockPos();
		s.cx = camera.x - pos.getX();
		s.cy = camera.y - pos.getY();
		s.cz = camera.z - pos.getZ();
		Level level = be.getLevel();
		if (level == null || !(be.getBlockState().getBlock() instanceof MatterBlock)) return;
		for (Direction d : Direction.values()) {
			BlockEntity n = level.getBlockEntity(pos.relative(d));
			if (n instanceof MatterBlockEntity m && covers(m.data(), s.data)) s.hidden |= 1 << d.ordinal();
			else if (s.data.clear()) s.depth[d.ordinal()] = bodyBehind(level, pos, d, s.data.material);
		}
	}

	/** This block plus the blocks of the same material in a straight line behind the face toward {@code d}. */
	private static int bodyBehind(Level level, BlockPos pos, Direction d, String material) {
		int k = 1;
		BlockPos p = pos.relative(d.getOpposite());
		while (k < MAX_BODY && level.getBlockEntity(p) instanceof MatterBlockEntity m && m.data().material.equals(material)) {
			k++;
			p = p.relative(d.getOpposite());
		}
		return k;
	}

	/** An opaque neighbour hides the face; so does the same material (no interface between them). */
	private static boolean covers(MatterData neighbour, MatterData me) {
		if (neighbour.isEmpty()) return false;
		return neighbour.opaque() || neighbour.material.equals(me.material);
	}

	@Override
	public void submit(State s, PoseStack pose, SubmitNodeCollector collector, CameraRenderState camera) {
		if (s.data.isEmpty()) return;
		RenderType type = RenderTypes.entityTranslucent(WHITE);
		collector.submitCustomGeometry(pose, type, (p, vc) -> cube(p, vc, s));
	}

	// each face: origin, u, v with u x v = the outward normal, so the quad winds counter-clockwise from outside
	private static final Direction[] FACE = {Direction.EAST, Direction.WEST, Direction.UP, Direction.DOWN, Direction.SOUTH, Direction.NORTH};
	private static final float[][] ORIGIN = {{1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
	private static final float[][] U = {{0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}};
	private static final float[][] V = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}};

	private static void cube(PoseStack.Pose pose, VertexConsumer vc, State s) {
		MatterData d = s.data;
		boolean clear = d.clear();
		float e = (float) Math.max(0, Math.min(1, d.emission));
		int r = mix(d.rgba[0], d.glowRgb[0], e), g = mix(d.rgba[1], d.glowRgb[1], e), b = mix(d.rgba[2], d.glowRgb[2], e);
		int a = d.rgba[3];
		int light = e > 0.05f ? FULL_BRIGHT : s.lightCoords;
		double env = Math.max((s.lightCoords >> 20) & 15, (s.lightCoords >> 4) & 15) / 15.0;   // the light around it, to reflect
		for (int f = 0; f < 6; f++) {
			Direction dir = FACE[f];
			if ((s.hidden & (1 << dir.ordinal())) != 0) continue;
			float[] o = ORIGIN[f], u = U[f], v = V[f];
			float nx = dir.getStepX(), ny = dir.getStepY(), nz = dir.getStepZ();
			int lf = light;
			if (clear) {
				double vx = s.cx - (o[0] + (u[0] + v[0]) / 2), vy = s.cy - (o[1] + (u[1] + v[1]) / 2), vz = s.cz - (o[2] + (u[2] + v[2]) / 2);
				double dist = Math.sqrt(vx * vx + vy * vy + vz * vz);
				double cos = (nx * vx + ny * vy + nz * vz) / dist;
				double[] ca = clearFace(d, Math.abs(cos), cos > 0, s.depth[dir.ordinal()], env);
				r = (int) ca[0]; g = (int) ca[1]; b = (int) ca[2]; a = (int) ca[3];
				if (a <= 0) continue;
				lf = FULL_BRIGHT;               // the colour already carries the light it reflects
				nx = 0; ny = 1; nz = 0;         // the entity shader's face shading is not physics: no darkening by direction
			}
			vertex(vc, pose, o[0], o[1], o[2], 0, 0, r, g, b, a, lf, nx, ny, nz);
			vertex(vc, pose, o[0] + u[0], o[1] + u[1], o[2] + u[2], 1, 0, r, g, b, a, lf, nx, ny, nz);
			vertex(vc, pose, o[0] + u[0] + v[0], o[1] + u[1] + v[1], o[2] + u[2] + v[2], 1, 1, r, g, b, a, lf, nx, ny, nz);
			vertex(vc, pose, o[0] + v[0], o[1] + v[1], o[2] + v[2], 0, 1, r, g, b, a, lf, nx, ny, nz);
		}
	}

	/**
	 * One face of a clear dielectric as (r, g, b, alpha) bytes. Composited over what is behind it, src-over gives
	 * F * env + (1 - F) * T * behind, which is the physics: F reflected, the rest transmitted and attenuated.
	 */
	static double[] clearFace(MatterData d, double cosI, boolean front, int blocksBehind, double env) {
		double n = d.n;
		double sinI2 = Math.max(0, 1 - cosI * cosI);
		double sinT = Math.sqrt(sinI2) / n;
		double cosT = Math.sqrt(Math.max(0, 1 - sinT * sinT));
		double F, T = 1;
		if (front) {                           // air -> material
			F = fresnel(1.0, n, cosI, cosT);
			double path = blocksBehind * d.pitchM / Math.max(cosT, 0.05);
			double sum = 0;
			for (int c = 0; c < 3; c++) sum += Math.exp(-d.absorption[c] * path);
			T = sum / 3;
		} else {                               // material -> air, the ray having come in at cosT
			F = fresnel(n, 1.0, cosT, cosI);
		}
		double alpha = 1 - (1 - F) * T;
		if (alpha <= 1e-4) return new double[] {0, 0, 0, 0};
		double c = Math.max(0, Math.min(1, F * env / alpha));
		return new double[] {Math.round(255 * c), Math.round(255 * c), Math.round(255 * c), Math.round(255 * Math.min(1, alpha))};
	}

	/** Unpolarised Fresnel reflectance from index n1 into n2, with cosines of the incident and transmitted angles. */
	static double fresnel(double n1, double n2, double cosI, double cosT) {
		double rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
		double rp = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT);
		return Math.min(1, (rs * rs + rp * rp) / 2);
	}

	private static int mix(int base, int glow, float e) {
		return Math.max(0, Math.min(255, Math.round(base * (1 - e) + glow * e)));
	}

	private static void vertex(VertexConsumer vc, PoseStack.Pose pose, float x, float y, float z, float u, float v,
							   int r, int g, int b, int a, int light, float nx, float ny, float nz) {
		vc.addVertex(pose, x, y, z).setColor(r, g, b, a).setUv(u, v).setOverlay(OverlayTexture.NO_OVERLAY)
			.setLight(light).setNormal(pose, nx, ny, nz);
	}
}
