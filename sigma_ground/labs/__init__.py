"""
sigma_ground.labs — Mini-lab simulation scene engine.

Build declarative physics simulations from natural descriptions.
Objects inherit ALL computable properties from sigma-ground's
material database via the material cascade.

Usage:
    from sigma_ground.labs import (
        SimulationScene, SimObject, SimEvent,
        Environment, Medium, LightSource,
        run_simulation,
    )

    scene = SimulationScene(
        name='free_fall',
        description='Iron ball dropped from 1 meter',
        objects=[
            SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                      position=(0, 1, 0)),
        ],
        environment=Environment(medium=Medium.vacuum()),
        duration=0.5,
    )

    result = run_simulation(scene)
    print(result.summary)
"""

from .scene import SimulationScene, SimObject, SimEvent
from .environment import (
    Environment, Medium, GroundConfig, BoundaryPlane, LightSource,
)
from .result import SimulationResult, Snapshot, ObjectState, CollisionEvent
from .runner import run_simulation
from .builder import cascade_material
from .validation import validate, KNOWN_SCENARIOS
