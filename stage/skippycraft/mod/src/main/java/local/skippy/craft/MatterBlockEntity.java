package local.skippy.craft;

import net.minecraft.core.BlockPos;
import net.minecraft.core.HolderLookup;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.game.ClientGamePacketListener;
import net.minecraft.network.protocol.game.ClientboundBlockEntityDataPacket;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.storage.ValueInput;
import net.minecraft.world.level.storage.ValueOutput;

/** The record a matter block carries, saved with the world and sent to the client that draws it. */
public class MatterBlockEntity extends BlockEntity {
	private MatterData data = MatterData.EMPTY;

	public MatterBlockEntity(BlockPos pos, BlockState state) {
		super(SkippyCraftMod.MATTER_ENTITY, pos, state);
	}

	public MatterData data() {
		return data;
	}

	public void setRecord(String json) {
		if (json.equals(data.json)) return;
		data = new MatterData(json);
		setChanged();
		if (level != null && !level.isClientSide()) level.sendBlockUpdated(worldPosition, getBlockState(), getBlockState(), 2);
	}

	@Override
	protected void saveAdditional(ValueOutput out) {
		super.saveAdditional(out);
		out.putString("record", data.json);
	}

	@Override
	protected void loadAdditional(ValueInput in) {
		super.loadAdditional(in);
		data = new MatterData(in.getStringOr("record", ""));
	}

	@Override
	public Packet<ClientGamePacketListener> getUpdatePacket() {
		return ClientboundBlockEntityDataPacket.create(this);
	}

	@Override
	public CompoundTag getUpdateTag(HolderLookup.Provider registries) {
		return saveCustomOnly(registries);
	}
}
