package local.skippy.craft;

import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.chat.Component;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.EntityBlock;
import net.minecraft.world.level.block.RenderShape;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.StateDefinition;
import net.minecraft.world.level.block.state.properties.IntegerProperty;
import net.minecraft.world.phys.BlockHitResult;

/**
 * The matter block: one cell of a physics world, whose material, mass and state live in its record and whose look
 * (colour, opacity, glow) is drawn from that record, not from a texture. Its light level is a block state, so
 * a glowing block really lights its neighbours.
 */
public class MatterBlock extends Block implements EntityBlock {
	public static final IntegerProperty LIGHT = IntegerProperty.create("light", 0, 15);

	public MatterBlock(Properties properties) {
		super(properties);
		registerDefaultState(stateDefinition.any().setValue(LIGHT, 0));
	}

	@Override
	protected void createBlockStateDefinition(StateDefinition.Builder<Block, BlockState> builder) {
		builder.add(LIGHT);
	}

	@Override
	public BlockEntity newBlockEntity(BlockPos pos, BlockState state) {
		return new MatterBlockEntity(pos, state);
	}

	@Override
	protected RenderShape getRenderShape(BlockState state) {
		return RenderShape.INVISIBLE;          // the block entity renderer draws it from the record
	}

	@Override
	protected boolean skipRendering(BlockState state, BlockState neighbour, Direction direction) {
		return false;
	}

	/** Right-click: the record, in one line, on the action bar ("what is that"). */
	@Override
	protected InteractionResult useWithoutItem(BlockState state, Level level, BlockPos pos, Player player, BlockHitResult hit) {
		if (!level.isClientSide() && level.getBlockEntity(pos) instanceof MatterBlockEntity be) {
			player.sendOverlayMessage(Component.literal(be.data().isEmpty() ? "empty matter block" : be.data().summary()));
		}
		return InteractionResult.SUCCESS;
	}
}
