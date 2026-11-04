package local.skippy.craft.test;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import local.skippy.craft.MatterBlock;
import local.skippy.craft.MatterBlockEntity;
import local.skippy.craft.SkippyCraftMod;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;

import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * A scripted player (spectator) in a private singleplayer world: places every material as a matter block on a platform
 * high in the air, flies the camera to it and screenshots it by day and by night. The screenshots are the verdict on how
 * the renderer draws a record: colour, opacity, glow, the one-pane volume, the light a hot block throws.
 */
public class MatterClientTest implements FabricClientGameTest {
	private static final int Y = 200;

	private JsonObject records() throws Exception {
		try (var in = getClass().getResourceAsStream("/records.json")) {
			return JsonParser.parseReader(new InputStreamReader(in, StandardCharsets.UTF_8)).getAsJsonObject();
		}
	}

	private static void put(ServerLevel level, int x, int y, int z, String json) {
		BlockPos pos = new BlockPos(x, y, z);
		int light = JsonParser.parseString(json).getAsJsonObject().get("light").getAsInt();
		level.setBlock(pos, SkippyCraftMod.MATTER_BLOCK.defaultBlockState().setValue(MatterBlock.LIGHT, light), 2 | 16);
		if (level.getBlockEntity(pos) instanceof MatterBlockEntity be) be.setRecord(json);
	}

	@Override
	public void runTest(ClientGameTestContext context) {
		try (TestSingleplayerContext world = context.worldBuilder().create()) {
			JsonObject rec = records();
			world.getServer().runOnServer(server -> {
				ServerLevel level = server.overworld();
				BlockState stone = Blocks.QUARTZ_BLOCK.defaultBlockState();
				for (int x = -8; x <= 8; x++) for (int z = -3; z <= 6; z++) level.setBlock(new BlockPos(x, Y - 1, z), stone, 2 | 16);
				// a back wall, so a see-through block has something behind it
				for (int x = -8; x <= 8; x++) for (int y = Y; y <= Y + 4; y++) level.setBlock(new BlockPos(x, y, 5), Blocks.OBSIDIAN.defaultBlockState(), 2 | 16);
				// the row: gold, oak, glass, a 3x3x3 water volume, warm tungsten, hot (moving) tungsten, and a stone for comparison
				put(level, -7, Y, 2, rec.get("gold").getAsString());
				put(level, -5, Y, 2, rec.get("oak").getAsString());
				put(level, -3, Y, 2, rec.get("glass").getAsString());
				for (int dx = 0; dx < 3; dx++) for (int dy = 0; dy < 3; dy++) for (int dz = 0; dz < 3; dz++) {
					put(level, -1 + dx, Y + dy, 1 + dz, rec.get("water").getAsString());
				}
				put(level, 3, Y, 2, rec.get("warm_tungsten").getAsString());
				put(level, 5, Y, 2, rec.get("hot_tungsten").getAsString());
				level.setBlock(new BlockPos(7, Y, 2), Blocks.STONE.defaultBlockState(), 2 | 16);
			});
			world.getServer().runCommand("gamemode spectator @a");
			world.getServer().runCommand("time set noon");
			world.getServer().runCommand("weather clear");
			world.getServer().runCommand("tp @a 0.5 " + (Y + 3) + " -4.5 0 12");
			context.waitTicks(40);
			context.waitTicks(60);
			context.takeScreenshot("day-row");
			world.getServer().runCommand("tp @a -5.5 " + (Y + 1.5) + " -1.5 -20 5");
			context.waitTicks(10);
			context.takeScreenshot("day-gold-oak-glass");
			world.getServer().runCommand("tp @a 1.5 " + (Y + 1.5) + " -2.5 0 0");
			context.waitTicks(10);
			context.takeScreenshot("day-water-volume");
			world.getServer().runCommand("tp @a 1.5 " + (Y + 0.4) + " -3.5 0 -2");
			context.waitTicks(10);
			context.takeScreenshot("day-water-grazing");
			world.getServer().runCommand("tp @a 1.5 " + (Y + 6) + " -1.5 0 60");
			context.waitTicks(10);
			context.takeScreenshot("day-water-from-above");
			world.getServer().runCommand("time set midnight");
			world.getServer().runCommand("tp @a 4.0 " + (Y + 2.5) + " -3.5 0 10");
			context.waitTicks(40);
			context.takeScreenshot("night-hot-tungsten");
			world.getServer().runCommand("tp @a 0.5 " + (Y + 3) + " -4.5 0 12");
			context.waitTicks(20);
			context.takeScreenshot("night-row");
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
