package local.skippy.craft.client.mixin;

import com.mojang.renderpearl.api.commands.RenderPass;
import local.skippy.craft.client.TintRenderer;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.chunk.ChunkSectionsToRender;
import net.minecraft.client.renderer.feature.FeatureRenderDispatcher;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Where the tint of clear matter is drawn: into the level's open solid pass, once the solid terrain and features are in and before
 * any translucent surface (the improved-transparency pass or the classic sorted one) is drawn over them. Fabric's own
 * after-solid-features event does not hand over the pass, and the engine allows only one open at a time.
 */
@Mixin(LevelRenderer.class)
public abstract class LevelRendererMixin {
	@Shadow
	private LevelRenderState levelRenderState;

	@Inject(method = "executeSolid", at = @At("RETURN"))
	private void skippycraft$tint(ChunkSectionsToRender sections, FeatureRenderDispatcher.PreparedFrame frame, RenderPass pass, CallbackInfo ci) {
		TintRenderer.draw(pass, levelRenderState.cameraRenderState);
	}
}
