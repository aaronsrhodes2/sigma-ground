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
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Future;

import io.papermc.paper.event.player.AsyncChatEvent;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.block.data.BlockData;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.Vector;

/**
 * SkippyCraft's block writer and its ear. Two frames on one socket, each answered in turn:
 *
 * BLOCKS (magic 'SKCF'): the theater service (stage/skippycraft/theater.py) sends each tick's changed blocks in one
 * binary frame, and this sets them all on the server thread with no block physics, then answers. RCON sends one command
 * per round trip; this takes tens of thousands of blocks per frame. Frame (big-endian): int magic; int palette size;
 * that many UTF strings (block data, e.g. "minecraft:blue_stained_glass"); int run count; that many runs of (int x,
 * int y, int z, short length, short palette index), each run along +x. Reply: int blocks set, int milliseconds.
 *
 * SAID (magic 'SKCP', P6: controls from inside the game): what the player said to Skippy since the last poll. A chat
 * line starting with "skippy" (any case; the word and any comma/colon after it stripped) or the command "/theater ..."
 * is queued with where he was and what he was looking at, read on the server thread. Reply: int count; per message:
 * UTF player, UTF text, double eye x y z, double look dx dy dz, byte hit, int block x y z (the block he looks at, up to
 * 64 m away; zeros when none).
 *
 * Listens on 0.0.0.0:25576 inside the container; compose publishes it on 127.0.0.1:42576 only.
 */
public final class SkippyCraftPlugin extends JavaPlugin implements Listener {
    private static final int MAGIC_BLOCKS = 0x534B4346;
    private static final int MAGIC_SAID = 0x534B4350;
    private static final int PORT = 25576;
    private ServerSocket server;
    private Thread acceptor;
    private final ConcurrentLinkedQueue<Said> said = new ConcurrentLinkedQueue<>();

    private record Run(int x, int y, int z, int length, int palette) {}

    private record Said(String player, String text, double x, double y, double z, double dx, double dy, double dz,
                        boolean hit, int bx, int by, int bz) {}

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
        getServer().getPluginManager().registerEvents(this, this);
        getLogger().info("block writer listening on " + PORT + "; say 'Skippy, ...' or /theater ...");
    }

    @Override
    public void onDisable() {
        try {
            if (server != null) server.close();
        } catch (IOException ignored) {
        }
    }

    // ── what he says ───────────────────────────────────────────────────────────────────────────────────────────────
    @EventHandler
    public void onChat(AsyncChatEvent e) {
        String text = PlainTextComponentSerializer.plainText().serialize(e.message()).trim();
        if (!text.toLowerCase().startsWith("skippy")) return;
        String rest = text.substring("skippy".length()).replaceFirst("^[\\s,:!.]+", "").trim();
        Player p = e.getPlayer();
        Bukkit.getScheduler().runTask(this, () -> queue(p, rest));   // the look target is read on the server thread
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player p)) {
            sender.sendMessage("/theater is for a player in the world");
            return true;
        }
        queue(p, String.join(" ", args).trim());
        return true;
    }

    private void queue(Player p, String text) {
        Location eye = p.getEyeLocation();
        Vector d = eye.getDirection();
        Block b = p.getTargetBlockExact(64);
        said.add(new Said(p.getName(), text, eye.getX(), eye.getY(), eye.getZ(), d.getX(), d.getY(), d.getZ(),
                          b != null, b == null ? 0 : b.getX(), b == null ? 0 : b.getY(), b == null ? 0 : b.getZ()));
    }

    // ── the socket ─────────────────────────────────────────────────────────────────────────────────────────────────
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
                    out.writeDouble(q.x()); out.writeDouble(q.y()); out.writeDouble(q.z());
                    out.writeDouble(q.dx()); out.writeDouble(q.dy()); out.writeDouble(q.dz());
                    out.writeBoolean(q.hit());
                    out.writeInt(q.bx()); out.writeInt(q.by()); out.writeInt(q.bz());
                }
                out.flush();
                continue;
            }
            if (magic != MAGIC_BLOCKS) throw new IOException("bad frame");
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
