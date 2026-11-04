package local.skippy.craft;

import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.message.v1.ServerMessageEvents;
import net.minecraft.commands.Commands;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.entity.BlockEntityType;
import net.minecraft.world.level.block.state.BlockBehaviour;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Set;

/**
 * SkippyCraft: a matter block (material, mass and state from sigma-ground, drawn from the record) and the bridge the
 * theater service (stage/skippycraft/theater.py) writes the world through and listens to. It replaces the Paper plugin.
 */
public class SkippyCraftMod implements ModInitializer {
	public static final String MOD_ID = "skippycraft";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	public static Block MATTER_BLOCK;
	public static BlockEntityType<MatterBlockEntity> MATTER_ENTITY;

	public static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath(MOD_ID, path);
	}

	@Override
	public void onInitialize() {
		Identifier id = id("matter");
		ResourceKey<Block> key = ResourceKey.create(Registries.BLOCK, id);
		MATTER_BLOCK = Registry.register(BuiltInRegistries.BLOCK, key, new MatterBlock(
			BlockBehaviour.Properties.of().setId(key).strength(0.5f).noOcclusion().noLootTable()
				.lightLevel(state -> state.getValue(MatterBlock.LIGHT))));
		MATTER_ENTITY = Registry.register(BuiltInRegistries.BLOCK_ENTITY_TYPE, id,
			new BlockEntityType<>(MatterBlockEntity::new, Set.of(MATTER_BLOCK)));

		SkippyBridge bridge = new SkippyBridge();
		ServerLifecycleEvents.SERVER_STARTED.register(server -> {
			if (server.isDedicatedServer()) bridge.start(server);
		});
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> bridge.stop());
		ServerMessageEvents.CHAT_MESSAGE.register((message, sender, params) -> {
			String text = message.decoratedContent().getString().trim();
			if (text.toLowerCase().startsWith("skippy")) {
				bridge.said(sender, text.substring("skippy".length()).replaceFirst("^[\\s,:!.]+", "").trim());
			}
		});
		CommandRegistrationCallback.EVENT.register((dispatcher, context, selection) -> dispatcher.register(
			Commands.literal("theater").then(Commands.argument("text", StringArgumentType.greedyString()).executes(ctx -> {
				bridge.said(ctx.getSource().getPlayerOrException(), StringArgumentType.getString(ctx, "text").trim());
				ctx.getSource().sendSuccess(() -> Component.literal("theater: heard"), false);
				return 1;
			}))));
		LOGGER.info("SkippyCraft: matter block {} registered", id);
	}
}
