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
 * Draws a matter block from its record. This class computes NO physical quantity: every number it draws with is read from the
 * tables sigma-ground's optics computed into the record (stage/skippycraft/matter.py): the Fresnel reflectance, the refraction
 * angle and the Beer-Lambert transmittance. What it does is geometry (the view angle of a face, the blocks behind it, where a
 * refracted ray lands) and compositing.
 *
 * An opaque material is a cube in its record's colour, brightened toward its glow colour by its emission, lit as bright as it
 * glows, and drawn where a refracted ray puts it when clear matter lies between it and the camera ({@link Refraction}).
 *
 * A clear dielectric is drawn the way light meets it: each face reflects the tabulated Fresnel fraction at the angle the camera
 * sees it; the rest goes through the body and is attenuated by the tabulated transmittance of the real path (the blocks of the
 * same material behind the face, at the tabulated refraction angle). Faces between blocks of one material are not drawn: there
 * is no interface. The environment a face reflects is sigma-ground's clear sky (or its ground, for a ray that points down)
 * along the ray mirrored at each corner of the face ({@link SkyLight}); a back face assumes parallel faces. Each approximation is
 * a named gap in KNOWN_GAPS.md.
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
		double[][] shift;                   // per corner (bit 0 x, bit 1 y, bit 2 z): where refraction moves it, or null
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
		s.shift = null;
		Level level = be.getLevel();
		if (level == null || !(be.getBlockState().getBlock() instanceof MatterBlock)) return;
		if (s.data.opaque() && Refraction.enabled()) s.shift = Refraction.corners(level, camera, pos);
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

	/** A theater is dozens of blocks across (one cell, one block): far more than the default 64 blocks a block entity is drawn at. */
	@Override
	public int getViewDistance() {
		return 256;
	}

	@Override
	public void submit(State s, PoseStack pose, SubmitNodeCollector collector, CameraRenderState camera) {
		if (s.data.isEmpty()) return;
		if (s.data.clear()) tint(s);
		RenderType type = RenderTypes.entityTranslucent(WHITE);
		collector.submitCustomGeometry(pose, type, (p, vc) -> cube(p, vc, s));
	}

	/**
	 * The colour a clear body takes out of the light behind each face that looks at the camera: the face's per-channel
	 * transmittance, tabulated by sigma-ground for the path through the body, multiplied into the scene by {@link TintRenderer}.
	 */
	private static void tint(State s) {
		MatterData d = s.data;
		for (int f = 0; f < 6; f++) {
			Direction dir = FACE[f];
			if ((s.hidden & (1 << dir.ordinal())) != 0) continue;
			float[] o = ORIGIN[f], u = U[f], v = V[f];
			double vx = s.cx - (o[0] + (u[0] + v[0]) / 2), vy = s.cy - (o[1] + (u[1] + v[1]) / 2), vz = s.cz - (o[2] + (u[2] + v[2]) / 2);
			double cos = (dir.getStepX() * vx + dir.getStepY() * vy + dir.getStepZ() * vz) / Math.sqrt(vx * vx + vy * vy + vz * vz);
			if (cos <= 0) continue;                                       // a face turned away: the light has not yet come in through it
			double cosT = d.tables.cosRefractedIn(Math.min(1, cos));
			double[] rgb = d.tables.transmittanceRgb(s.depth[dir.ordinal()] / Math.max(cosT, 0.05));
			float[] c = new float[12];
			for (int k = 0; k < 4; k++) {
				float a = (k == 1 || k == 2) ? 1 : 0, b = (k >= 2) ? 1 : 0;
				c[3 * k] = (float) (o[0] + u[0] * a + v[0] * b - s.cx);
				c[3 * k + 1] = (float) (o[1] + u[1] * a + v[1] * b - s.cy);
				c[3 * k + 2] = (float) (o[2] + u[2] * a + v[2] * b - s.cz);
			}
			TintRenderer.add(c, rgb);
		}
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
		double env = Math.max((s.lightCoords >> 20) & 15, (s.lightCoords >> 4) & 15) / 15.0;   // the light around it, when there is no sky table
		double skyExposure = ((s.lightCoords >> 20) & 15) / 15.0;                              // how much sky this block sees: the game's own sky light
		for (int f = 0; f < 6; f++) {
			Direction dir = FACE[f];
			if ((s.hidden & (1 << dir.ordinal())) != 0) continue;
			float[] o = ORIGIN[f], u = U[f], v = V[f];
			float nx = dir.getStepX(), ny = dir.getStepY(), nz = dir.getStepZ();
			int lf = light;
			double[][] cv = null;               // a clear face: r, g, b, alpha at each of its four corners
			if (clear) {
				cv = new double[4][];
				boolean any = false;
				for (int k = 0; k < 4; k++) {
					float ka = (k == 1 || k == 2) ? 1 : 0, kb = (k >= 2) ? 1 : 0;
					double vx = s.cx - (o[0] + u[0] * ka + v[0] * kb), vy = s.cy - (o[1] + u[1] * ka + v[1] * kb), vz = s.cz - (o[2] + u[2] * ka + v[2] * kb);
					double dist = Math.sqrt(vx * vx + vy * vy + vz * vz);
					double cos = (nx * vx + ny * vy + nz * vz) / dist;
					// the ray from the camera mirrors off the face: r = -v + 2 (v . n) n, v the unit vector from the face to the camera
					double rx = -vx / dist + 2 * cos * nx, ry = -vy / dist + 2 * cos * ny, rz = -vz / dist + 2 * cos * nz;
					// the free surface of water, seen from above, is wind-roughened: its table adds the sun's glint to the sky it mirrors
					double[] sheet = dir == Direction.UP && cos > 0 && d.material.equals(SkyLight.waterMaterial())
						? SkyLight.water(rx, ry, rz, skyExposure) : null;
					cv[k] = sheet != null ? new double[] {Math.round(sheet[0]), Math.round(sheet[1]), Math.round(sheet[2]), Math.round(sheet[3])}
						: clearFace(d, Math.abs(cos), cos > 0, env, SkyLight.reflected(rx, ry, rz, skyExposure));
					any |= cv[k][3] > 0;
				}
				if (!any) continue;
				lf = FULL_BRIGHT;               // the colour already carries the light it reflects
				nx = 0; ny = 1; nz = 0;         // the entity shader's face shading is not physics: no darkening by direction
			}
			for (int k = 0; k < 4; k++) {
				float ka = (k == 1 || k == 2) ? 1 : 0, kb = (k >= 2) ? 1 : 0;
				int cr = r, cg = g, cb = b, ca = a;
				if (cv != null) {
					cr = (int) cv[k][0]; cg = (int) cv[k][1]; cb = (int) cv[k][2]; ca = (int) cv[k][3];
				}
				corner(vc, pose, s, o[0] + u[0] * ka + v[0] * kb, o[1] + u[1] * ka + v[1] * kb, o[2] + u[2] * ka + v[2] * kb,
					ka, kb, cr, cg, cb, ca, lf, nx, ny, nz);
			}
		}
	}

	/**
	 * One face of a clear dielectric as (r, g, b, alpha) bytes. Composited over what is behind it, src-over gives
	 * R * env + (1 - R) * T * behind: R reflected, the rest transmitted and attenuated, with R, the refraction angle and T read
	 * from sigma-ground's tables.
	 */
	static double[] clearFace(MatterData d, double cosI, boolean front, double env, double[] mirror) {
		MatterData.Tables t = d.tables;
		double R = front ? t.reflectanceIn(cosI)                           // air -> material
			: t.reflectanceOut(t.cosRefractedIn(cosI));                    // material -> air, the ray having come in at cosT
		if (R <= 1e-4) return new double[] {0, 0, 0, 0};
		double alpha = Math.round(255 * Math.min(1, R));
		if (mirror != null) return new double[] {Math.round(mirror[0]), Math.round(mirror[1]), Math.round(mirror[2]), alpha};
		double c = Math.round(255 * Math.max(0, Math.min(1, env)));        // no sky table: white at the local light
		return new double[] {c, c, c, alpha};
	}

	/** A cube corner (each coordinate 0 or 1), moved by where refraction puts it. */
	private static void corner(VertexConsumer vc, PoseStack.Pose pose, State s, float x, float y, float z, float u, float v,
							   int r, int g, int b, int a, int light, float nx, float ny, float nz) {
		if (s.shift != null) {
			double[] d = s.shift[(x > 0.5f ? 1 : 0) | (y > 0.5f ? 2 : 0) | (z > 0.5f ? 4 : 0)];
			x += (float) d[0];
			y += (float) d[1];
			z += (float) d[2];
		}
		vertex(vc, pose, x, y, z, u, v, r, g, b, a, light, nx, ny, nz);
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
