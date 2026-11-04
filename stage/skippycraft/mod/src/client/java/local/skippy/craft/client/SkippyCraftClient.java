package local.skippy.craft.client;

import local.skippy.craft.SkippyCraftMod;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.BlockEntityRendererRegistry;

public class SkippyCraftClient implements ClientModInitializer {
	@Override
	public void onInitializeClient() {
		BlockEntityRendererRegistry.register(SkippyCraftMod.MATTER_ENTITY, ctx -> new MatterRenderer());
	}
}
