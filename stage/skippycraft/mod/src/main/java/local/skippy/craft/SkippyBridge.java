package local.skippy.craft;

import net.minecraft.commands.arguments.blocks.BlockStateParser;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * The theater service's door into the world (it replaces the Paper plugin; the frames are the same, plus one).
 * Three frames on one socket, each answered in turn (big-endian):
 *
 * BLOCKS (magic 'SKCF'): int palette size; that many UTF block states ("minecraft:gold_block"); int run count; runs of
 * (int x, int y, int z, short length, short palette index) along +x. Set with no block physics. Reply: int set, int ms.
 *
 * MATTER (magic 'SKCM'): int count; per block int x, int y, int z, int length n, n bytes of UTF-8 JSON (the record,
 * stage/skippycraft/matter.py; n = 0 clears the cell). Each becomes a skippycraft:matter block carrying that record, its
 * light level set from the record. Reply: int set, int ms.
 *
 * MATTER RUNS (magic 'SKCN'): the same, as the theater draws: int palette size; that many records (int length, that many
 * bytes of UTF-8 JSON; length 0 is air); int run count; runs of (int x, int y, int z, short length, short palette index) along
 * +x. A whole body of one material is one record and a few runs. Reply: int set, int ms.
 *
 * SAID (magic 'SKCP'): what the player said to Skippy since the last poll ("Skippy, ..." in chat or /theater ...), with
 * where he was and what he looked at: int count; per message UTF player, UTF text, double eye x y z, double look
 * dx dy dz, byte hit, int block x y z.
 *
 * Listens on 0.0.0.0:25576 inside the container; compose publishes it on 127.0.0.1:42576 only.
 */
public final class SkippyBridge {
	private static final int MAGIC_BLOCKS = 0x534B4346;
	private static final int MAGIC_MATTER = 0x534B434D;
	private static final int MAGIC_MATTER_RUNS = 0x534B434E;
	private static final int MAGIC_SAID = 0x534B4350;
	private static final int PORT = 25576;
	private static final int FLAGS = 2 | 16;        // send to clients, skip neighbour shape updates: no block physics

	private MinecraftServer server;
	private ServerSocket socket;
	private final ConcurrentLinkedQueue<Said> said = new ConcurrentLinkedQueue<>();

	private record Run(int x, int y, int z, int length, int palette) {}

	private record Cell(int x, int y, int z, String json) {}

	private record Said(String player, String text, Vec3 eye, Vec3 look, boolean hit, BlockPos block) {}

	public void start(MinecraftServer server) {
		this.server = server;
		try {
			socket = new ServerSocket();
			socket.setReuseAddress(true);
			socket.bind(new InetSocketAddress("0.0.0.0", PORT));
		} catch (IOException e) {
			SkippyCraftMod.LOGGER.error("block writer could not listen on {}: {}", PORT, e.toString());
			return;
		}
		Thread t = new Thread(this::accept, "skippycraft-bridge");
		t.setDaemon(true);
		t.start();
		SkippyCraftMod.LOGGER.info("block writer listening on {}; say 'Skippy, ...' or /theater ...", PORT);
	}

	public void stop() {
		try {
			if (socket != null) socket.close();
		} catch (IOException ignored) {
		}
	}

	/** Called on the server thread, so the look target is read where the world may be read. */
	public void said(ServerPlayer p, String text) {
		Vec3 eye = p.getEyePosition();
		Vec3 look = p.getLookAngle();
		HitResult h = p.pick(64.0, 1.0f, false);
		boolean hit = h.getType() == HitResult.Type.BLOCK;
		said.add(new Said(p.getName().getString(), text, eye, look, hit, hit ? ((BlockHitResult) h).getBlockPos() : BlockPos.ZERO));
	}

	private void accept() {
		while (!socket.isClosed()) {
			try (Socket s = socket.accept()) {
				s.setTcpNoDelay(true);
				serve(s);
			} catch (IOException e) {
				if (!socket.isClosed()) SkippyCraftMod.LOGGER.warn("block writer: {}", e.toString());
			}
		}
	}

	private void serve(Socket s) throws IOException {
		DataInputStream in = new DataInputStream(new BufferedInputStream(s.getInputStream(), 1 << 16));
		DataOutputStream out = new DataOutputStream(new BufferedOutputStream(s.getOutputStream()));
		while (true) {
			int magic;
			try {
				magic = in.readInt();
			} catch (EOFException e) {
				return;
			}
			if (magic == MAGIC_SAID) {
				List<Said> batch = new ArrayList<>();
				Said m;
				while ((m = said.poll()) != null) batch.add(m);
				out.writeInt(batch.size());
				for (Said q : batch) {
					out.writeUTF(q.player());
					out.writeUTF(q.text());
					out.writeDouble(q.eye().x); out.writeDouble(q.eye().y); out.writeDouble(q.eye().z);
					out.writeDouble(q.look().x); out.writeDouble(q.look().y); out.writeDouble(q.look().z);
					out.writeBoolean(q.hit());
					out.writeInt(q.block().getX()); out.writeInt(q.block().getY()); out.writeInt(q.block().getZ());
				}
				out.flush();
				continue;
			}
			int[] result;
			if (magic == MAGIC_BLOCKS) {
				int np = in.readInt();
				List<String> names = new ArrayList<>(np);
				for (int i = 0; i < np; i++) names.add(in.readUTF());
				int nr = in.readInt();
				List<Run> runs = new ArrayList<>(nr);
				for (int i = 0; i < nr; i++) runs.add(new Run(in.readInt(), in.readInt(), in.readInt(), in.readShort(), in.readShort()));
				result = onServerThread(() -> applyBlocks(names, runs));
			} else if (magic == MAGIC_MATTER) {
				int n = in.readInt();
				List<Cell> cells = new ArrayList<>(n);
				for (int i = 0; i < n; i++) {
					int x = in.readInt(), y = in.readInt(), z = in.readInt();
					byte[] b = new byte[in.readInt()];
					in.readFully(b);
					cells.add(new Cell(x, y, z, new String(b, StandardCharsets.UTF_8)));
				}
				result = onServerThread(() -> applyMatter(cells));
			} else if (magic == MAGIC_MATTER_RUNS) {
				int np = in.readInt();
				List<String> records = new ArrayList<>(np);
				for (int i = 0; i < np; i++) {
					byte[] b = new byte[in.readInt()];
					in.readFully(b);
					records.add(new String(b, StandardCharsets.UTF_8));
				}
				int nr = in.readInt();
				List<Run> runs = new ArrayList<>(nr);
				for (int i = 0; i < nr; i++) runs.add(new Run(in.readInt(), in.readInt(), in.readInt(), in.readShort(), in.readShort()));
				result = onServerThread(() -> applyMatterRuns(records, runs));
			} else {
				throw new IOException("bad frame");
			}
			out.writeInt(result[0]);
			out.writeInt(result[1]);
			out.flush();
		}
	}

	private int[] onServerThread(java.util.function.Supplier<int[]> job) throws IOException {
		try {
			return server.submit(job).get();
		} catch (Exception e) {
			throw new IOException("apply failed: " + e);
		}
	}

	private int[] applyBlocks(List<String> names, List<Run> runs) {
		long t0 = System.nanoTime();
		ServerLevel level = server.overworld();
		BlockState[] palette = new BlockState[names.size()];
		for (int i = 0; i < palette.length; i++) {
			try {
				palette[i] = BlockStateParser.parseForBlock(BuiltInRegistries.BLOCK, names.get(i), false).blockState();
			} catch (Exception e) {
				SkippyCraftMod.LOGGER.warn("unknown block state {}: {}", names.get(i), e.toString());
				palette[i] = Blocks.AIR.defaultBlockState();
			}
		}
		int n = 0;
		BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
		for (Run r : runs) {
			BlockState state = palette[r.palette()];
			for (int dx = 0; dx < r.length(); dx++) {
				level.setBlock(p.set(r.x() + dx, r.y(), r.z()), state, FLAGS);
				n++;
			}
		}
		return new int[] {n, (int) ((System.nanoTime() - t0) / 1_000_000)};
	}

	private int[] applyMatter(List<Cell> cells) {
		long t0 = System.nanoTime();
		ServerLevel level = server.overworld();
		int n = 0;
		for (Cell c : cells) {
			if (placeMatter(level, new BlockPos(c.x(), c.y(), c.z()), c.json())) n++;
		}
		return new int[] {n, (int) ((System.nanoTime() - t0) / 1_000_000)};
	}

	private int[] applyMatterRuns(List<String> records, List<Run> runs) {
		long t0 = System.nanoTime();
		ServerLevel level = server.overworld();
		int n = 0;
		BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
		for (Run r : runs) {
			String json = records.get(r.palette());
			for (int dx = 0; dx < r.length(); dx++) {
				if (placeMatter(level, p.set(r.x() + dx, r.y(), r.z()), json)) n++;
			}
		}
		return new int[] {n, (int) ((System.nanoTime() - t0) / 1_000_000)};
	}

	/** Make {@code pos} a matter block carrying {@code json} (its light level set from the record), or air when json is empty. */
	public static boolean placeMatter(ServerLevel level, BlockPos pos, String json) {
		if (json.isEmpty()) {
			level.setBlock(pos, Blocks.AIR.defaultBlockState(), FLAGS);
			return true;
		}
		MatterData d;
		try {
			d = MatterData.of(json);
		} catch (Exception e) {
			SkippyCraftMod.LOGGER.warn("bad matter record at {}: {}", pos, e.toString());
			return false;
		}
		BlockState state = SkippyCraftMod.MATTER_BLOCK.defaultBlockState().setValue(MatterBlock.LIGHT, Math.max(0, Math.min(15, d.light)))
			.setValue(MatterBlock.CLEAR, d.clear());
		level.setBlock(pos.immutable(), state, FLAGS);
		if (level.getBlockEntity(pos) instanceof MatterBlockEntity m) m.setRecord(json);
		return true;
	}
}
