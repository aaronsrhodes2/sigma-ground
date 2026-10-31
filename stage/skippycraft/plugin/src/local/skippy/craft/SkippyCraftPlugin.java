package local.skippy.craft;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;

import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.block.data.BlockData;
import org.bukkit.plugin.java.JavaPlugin;

/**
 * SkippyCraft's block writer: the theater service (stage/skippycraft/theater.py) sends each tick's changed blocks here
 * in one binary frame, and this sets them all on the server thread with no block physics, then answers. RCON sends one
 * command per round trip; this takes tens of thousands of blocks per frame.
 *
 * Frame (big-endian): int magic 'SKCF'; int palette size; that many UTF strings (block data, e.g.
 * "minecraft:blue_stained_glass"); int run count; that many runs of (int x, int y, int z, short length, short palette
 * index), each run along +x. Reply: int blocks set, int milliseconds the server thread spent.
 *
 * Listens on 0.0.0.0:25576 inside the container; compose publishes it on 127.0.0.1:42576 only.
 */
public final class SkippyCraftPlugin extends JavaPlugin {
    private static final int MAGIC = 0x534B4346;
    private static final int PORT = 25576;
    private ServerSocket server;
    private Thread acceptor;

    @Override
    public void onEnable() {
        try {
            server = new ServerSocket();
            server.setReuseAddress(true);
            server.bind(new InetSocketAddress("0.0.0.0", PORT));
        } catch (IOException e) {
            getLogger().severe("block writer could not listen on " + PORT + ": " + e);
            return;
        }
        acceptor = new Thread(this::accept, "skippycraft-blocks");
        acceptor.setDaemon(true);
        acceptor.start();
        getLogger().info("block writer listening on " + PORT);
    }

    @Override
    public void onDisable() {
        try {
            if (server != null) server.close();
        } catch (IOException ignored) {
        }
    }

    private void accept() {
        while (!server.isClosed()) {
            try (Socket s = server.accept()) {
                s.setTcpNoDelay(true);
                serve(s);
            } catch (IOException e) {
                if (!server.isClosed()) getLogger().warning("block writer: " + e);
            }
        }
    }

    private record Run(int x, int y, int z, int length, int palette) {}

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
            if (magic != MAGIC) throw new IOException("bad frame");
            int np = in.readInt();
            List<String> names = new ArrayList<>(np);
            for (int i = 0; i < np; i++) names.add(in.readUTF());
            int nr = in.readInt();
            List<Run> runs = new ArrayList<>(nr);
            for (int i = 0; i < nr; i++)
                runs.add(new Run(in.readInt(), in.readInt(), in.readInt(), in.readShort(), in.readShort()));
            Future<int[]> done = Bukkit.getScheduler().callSyncMethod(this, () -> apply(names, runs));
            int[] result;
            try {
                result = done.get();
            } catch (Exception e) {
                throw new IOException("apply failed: " + e);
            }
            out.writeInt(result[0]);
            out.writeInt(result[1]);
            out.flush();
        }
    }

    private int[] apply(List<String> names, List<Run> runs) {
        long t0 = System.nanoTime();
        World world = Bukkit.getWorlds().get(0);
        BlockData[] palette = new BlockData[names.size()];
        for (int i = 0; i < palette.length; i++) palette[i] = Bukkit.createBlockData(names.get(i));
        int n = 0;
        for (Run r : runs) {
            BlockData data = palette[r.palette()];
            for (int dx = 0; dx < r.length(); dx++) {
                world.getBlockAt(r.x() + dx, r.y(), r.z()).setBlockData(data, false);
                n++;
            }
        }
        return new int[] {n, (int) ((System.nanoTime() - t0) / 1_000_000)};
    }
}
