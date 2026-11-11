package local.skippy.craft.test;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import local.skippy.craft.MatterBlock;
import local.skippy.craft.MatterBlockEntity;
import local.skippy.craft.SkippyBridge;
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
		SkippyBridge.placeMatter(level, new BlockPos(x, y, z), json);
	}

	/** The real slosh theater, drawn in matter mode at two sim times (stage/skippycraft/mod/gen_slosh_frames.py). */
	private void sloshFrames(ClientGameTestContext context, TestSingleplayerContext world) throws Exception {
		JsonArray frames;
		try (var in = getClass().getResourceAsStream("/slosh_frames.json")) {
			frames = JsonParser.parseReader(new InputStreamReader(in, StandardCharsets.UTF_8)).getAsJsonArray();
		}
		world.getServer().runCommand("time set noon");
		for (int f = 0; f < frames.size(); f++) {
			JsonObject fr = frames.get(f).getAsJsonObject();
			int xoff = 200 * (f + 1);
			JsonArray bb = fr.getAsJsonArray("bbox");
			int x0 = bb.get(0).getAsInt() + xoff, y0 = bb.get(1).getAsInt(), z0 = bb.get(2).getAsInt();
			int x1 = bb.get(3).getAsInt() + xoff, z1 = bb.get(5).getAsInt();
			JsonArray records = fr.getAsJsonArray("records");
			JsonArray runs = fr.getAsJsonArray("runs");
			world.getServer().runOnServer(server -> {
				ServerLevel level = server.overworld();
				for (int x = x0 - 110; x <= x1 + 110; x++) for (int z = z0 - 110; z <= z1 + 6; z++) level.setBlock(new BlockPos(x, y0 - 1, z), Blocks.QUARTZ_BLOCK.defaultBlockState(), 2 | 16);
				for (int x = x0 - 110; x <= x1 + 110; x++) for (int y = y0; y <= y0 + 50; y++) level.setBlock(new BlockPos(x, y, z1 + 5), Blocks.OBSIDIAN.defaultBlockState(), 2 | 16);
				for (var el : runs) {
					JsonArray r = el.getAsJsonArray();
					String json = records.get(r.get(4).getAsInt()).getAsString();
					for (int dx = 0; dx < r.get(3).getAsInt(); dx++) {
						SkippyBridge.placeMatter(level, new BlockPos(r.get(0).getAsInt() + xoff + dx, r.get(1).getAsInt(), r.get(2).getAsInt()), json);
					}
				}
			});
			double cx = (x0 + x1) / 2.0 + 0.5;
			String tag = fr.get("scene").getAsString() + "-t" + String.format("%.1f", fr.get("t").getAsDouble());
			double span = Math.max(x1 - x0, z1 - z0), dist = span * 0.6 + 8, up = span * 0.2 + 4;
			for (String mode : new String[] {"refracted", "straight"}) {      // the same views with refraction on, then off
				System.setProperty("skippycraft.refraction", mode.equals("refracted") ? "true" : "false");
				world.getServer().runCommand("tp @a " + cx + " " + (y0 + up) + " " + (z0 - dist) + " 0 6");
				context.waitTicks(60);
				context.takeScreenshot(tag + "-front-" + mode);
				world.getServer().runCommand("tp @a " + cx + " " + (y0 + up * 2.4) + " " + (z0 - dist * 0.5) + " 0 50");
				context.waitTicks(30);
				context.takeScreenshot(tag + "-above-" + mode);
				world.getServer().runCommand("tp @a " + (x0 - dist) + " " + (y0 + up * 0.6) + " " + ((z0 + z1) / 2.0) + " -90 4");
				context.waitTicks(40);
				context.takeScreenshot(tag + "-side-" + mode);
			}
			System.setProperty("skippycraft.refraction", "true");
		}
	}

	/**
	 * Three metres of water seen end-on against a white wall (60 cells of the 5 cm pitch): sigma-ground's absorption takes red out
	 * faster than green and blue (T = 0.41, 0.81, 0.94), so the part of the wall behind the water turns blue-green and the part
	 * beside it stays white. A short volume cannot show this: 15 cm of water absorbs 4% of red.
	 */
	private void deepWater(ClientGameTestContext context, TestSingleplayerContext world, String water) {
		world.getServer().runOnServer(server -> {
			ServerLevel level = server.overworld();
			for (int x = 30; x <= 54; x++) for (int y = 190; y <= 210; y++) level.setBlock(new BlockPos(x, y, 70), Blocks.QUARTZ_BLOCK.defaultBlockState(), 2 | 16);
			for (int dx = 0; dx < 3; dx++) for (int dy = 0; dy < 3; dy++) for (int dz = 0; dz < 60; dz++) {
				put(level, 41 + dx, Y + dy, 4 + dz, water);
			}
		});
		world.getServer().runCommand("time set noon");
		world.getServer().runCommand("tp @a 42.5 " + (Y + 1.5) + " 1.5 0 0");
		context.waitTicks(80);
		context.takeScreenshot("deep-water-tint");
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
			deepWater(context, world, rec.get("water").getAsString());
			sloshFrames(context, world);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
