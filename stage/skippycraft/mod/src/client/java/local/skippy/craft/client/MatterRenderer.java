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
import net.minecraft.core.Direction;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.phys.Vec3;

/**
 * Draws a matter block from its record: a cube in the record's colour at the record's opacity, brightened toward the
 * record's glow colour by its emission, and lit as bright as it glows. A face is not drawn where the neighbour is a
 * matter block that already covers it (an opaque one, or one drawn identically), so a volume of water is one pane,
 * not a grid.
 */
public final class MatterRenderer implements BlockEntityRenderer<MatterBlockEntity, MatterRenderer.State> {
	private static final Identifier WHITE = SkippyCraftMod.id("textures/misc/white.png");
	private static final int FULL_BRIGHT = 0xF000F0;

	public static final class State extends BlockEntityRenderState {
		MatterData data = MatterData.EMPTY;
		int hidden;      // bit i set: the face toward Direction.values()[i] is covered
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
		if (be.getLevel() == null) return;
		for (Direction d : Direction.values()) {
			if (be.getBlockState().getBlock() instanceof MatterBlock) {
				BlockEntity n = be.getLevel().getBlockEntity(be.getBlockPos().relative(d));
				if (n instanceof MatterBlockEntity m && covers(m.data(), s.data)) s.hidden |= 1 << d.ordinal();
			}
		}
	}

	private static boolean covers(MatterData neighbour, MatterData me) {
		if (neighbour.isEmpty()) return false;
		return neighbour.opaque() || java.util.Arrays.equals(neighbour.rgba, me.rgba);
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
		float e = (float) Math.max(0, Math.min(1, d.emission));
		int r = mix(d.rgba[0], d.glowRgb[0], e), g = mix(d.rgba[1], d.glowRgb[1], e), b = mix(d.rgba[2], d.glowRgb[2], e);
		int a = d.rgba[3];
		int light = e > 0.05f ? FULL_BRIGHT : s.lightCoords;
		for (int f = 0; f < 6; f++) {
			Direction dir = FACE[f];
			if ((s.hidden & (1 << dir.ordinal())) != 0) continue;
			float[] o = ORIGIN[f], u = U[f], v = V[f];
			float nx = dir.getStepX(), ny = dir.getStepY(), nz = dir.getStepZ();
			vertex(vc, pose, o[0], o[1], o[2], 0, 0, r, g, b, a, light, nx, ny, nz);
			vertex(vc, pose, o[0] + u[0], o[1] + u[1], o[2] + u[2], 1, 0, r, g, b, a, light, nx, ny, nz);
			vertex(vc, pose, o[0] + u[0] + v[0], o[1] + u[1] + v[1], o[2] + u[2] + v[2], 1, 1, r, g, b, a, light, nx, ny, nz);
			vertex(vc, pose, o[0] + v[0], o[1] + v[1], o[2] + v[2], 0, 1, r, g, b, a, light, nx, ny, nz);
		}
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
