package local.skippy.craft.client;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.BufferBuilder;
import com.mojang.blaze3d.vertex.ByteBufferBuilder;
import com.mojang.blaze3d.vertex.DefaultVertexFormat;
import com.mojang.blaze3d.vertex.MeshData;
import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.pipeline.BlendFactor;
import com.mojang.renderpearl.api.pipeline.BlendFunction;
import com.mojang.renderpearl.api.pipeline.ColorTargetState;
import com.mojang.renderpearl.api.pipeline.CompareOp;
import com.mojang.renderpearl.api.pipeline.DepthStencilState;
import com.mojang.renderpearl.api.pipeline.PrimitiveTopology;
import com.mojang.renderpearl.api.pipeline.RenderPipeline;
import com.mojang.renderpearl.api.pipeline.ShaderType;
import local.skippy.craft.SkippyCraftMod;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.client.renderer.state.level.CameraRenderState;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The colour a clear material takes out of the light that passes through it: the scene behind a clear face is multiplied by the
 * face's transmittance in each of R, G, B (blend: destination * source colour). Minecraft 26.3 sends every blended draw through its
 * order-independent translucency, whose transmittance is one number per pixel, so a per-channel multiply cannot ride on it; this
 * draws after the solid features, in the same open pass (see LevelRendererMixin), and before the translucent ones, and the reflected light is still composited over it
 * by the usual translucent pass (out = R * env + (1 - R) * T * behind).
 *
 * It computes no physics: each face arrives with the transmittance sigma-ground's optics tabulated for the path through the body
 * ({@link local.skippy.craft.MatterData.Tables#transmittanceRgb}); this only multiplies.
 */
public final class TintRenderer {
	private static final List<float[]> QUADS = new ArrayList<>();      // 4 corners (camera-relative x y z) then r g b in 0..1
	private static final int MAX_QUADS = 120_000;
	private static RenderPipeline pipeline;

	private TintRenderer() {}

	/** One face to tint: its four corners relative to the camera, and the transmittance per channel. */
	static void add(float[] corners, double[] rgb) {
		if (QUADS.size() >= MAX_QUADS) QUADS.clear();                   // a frame that never drew must not grow without end
		float[] q = new float[15];
		System.arraycopy(corners, 0, q, 0, 12);
		q[12] = (float) rgb[0];
		q[13] = (float) rgb[1];
		q[14] = (float) rgb[2];
		QUADS.add(q);
	}

	private static RenderPipeline pipeline() {
		if (pipeline == null) {
			RenderPipeline base = RenderPipelines.DEBUG_QUADS;           // position + colour, no texture: its shaders, uniforms and defines
			RenderPipeline.Builder b = RenderPipeline.builder()
				.withLocation(SkippyCraftMod.id("pipeline/tint_multiply"))
				.withVertexShader(base.getShaders().get(ShaderType.VERTEX))
				.withFragmentShader(base.getShaders().get(ShaderType.FRAGMENT))
				.withVertexBinding(0, DefaultVertexFormat.POSITION_COLOR)
				.withPrimitiveTopology(PrimitiveTopology.QUADS)
				.withCull(false)
				.withDepthStencilState(new DepthStencilState(base.getDepthStencilState() == null ? CompareOp.LESS_THAN_OR_EQUAL : base.getDepthStencilState().depthTest(), false))
				.withColorTargetState(new ColorTargetState(Optional.of(new BlendFunction(BlendFactor.ZERO, BlendFactor.SRC_COLOR)),
					base.getColorTargetStates().get(0).format(), ColorTargetState.WRITE_COLOR));
			for (var layout : base.getBindGroupLayouts()) b.withBindGroupLayout(layout);
			for (String flag : base.getShaderDefines().flags()) b.withShaderDefine(flag);
			for (var e : base.getShaderDefines().values().entrySet()) {
				try {
					b.withShaderDefine(e.getKey(), Integer.parseInt(e.getValue()));
				} catch (NumberFormatException ex) {
					b.withShaderDefine(e.getKey(), Float.parseFloat(e.getValue()));
				}
			}
			pipeline = b.build();
		}
		return pipeline;
	}

	public static void draw(RenderPass pass, CameraRenderState cam) {
		if (QUADS.isEmpty()) return;
		try {
			int vertexSize = DefaultVertexFormat.POSITION_COLOR.getVertexSize();
			try (ByteBufferBuilder bytes = ByteBufferBuilder.exactlySized(QUADS.size() * 4 * vertexSize)) {
				BufferBuilder bb = new BufferBuilder(bytes, PrimitiveTopology.QUADS, DefaultVertexFormat.POSITION_COLOR);
				for (float[] q : QUADS) {
					int r = Math.round(255 * q[12]), g = Math.round(255 * q[13]), b = Math.round(255 * q[14]);
					for (int i = 0; i < 4; i++) bb.addVertex(q[3 * i], q[3 * i + 1], q[3 * i + 2]).setColor(r, g, b, 255);
				}
				try (MeshData mesh = bb.buildOrThrow()) {
					int indexCount = mesh.drawState().indexCount();
					var device = RenderSystem.getDevice();
					GpuBuffer vertices = device.createBuffer(() -> "skippycraft tint vertices", GpuBuffer.USAGE_VERTEX | GpuBuffer.USAGE_COPY_DST, mesh.vertexBuffer());
					var indices = RenderSystem.getSequentialBuffer(PrimitiveTopology.QUADS);
					pass.setUniform("DynamicTransforms", RenderSystem.getDynamicUniforms().writeTransform(cam.viewRotationMatrix));
					pass.setPipeline(RenderSystem.getCompiledPipeline(pipeline()));
					pass.setVertexBuffer(0, vertices.slice());
					pass.setIndexBuffer(indices.getBuffer(indexCount), indices.type());
					pass.drawIndexed(indexCount, 1, 0, 0, 0);
					vertices.close();
				}
			}
		} finally {
			QUADS.clear();
		}
	}
}
