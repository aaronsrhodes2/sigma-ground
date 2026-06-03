"""
Simulation scene — declarative description of WHAT to simulate.

A SimulationScene is a data-only object. It says:
  "Here are the objects, here is the environment, here is what I care about."

It does NOT know how to run itself. That's the builder + runner's job.
This separation lets us validate, serialize, and inspect scenes without
importing the dynamics engine.

Dataclasses only — no physics imports.
"""


class SimObject:
    """An object in the simulation.

    Args:
        name:        Human-readable label ("baseball", "bat").
        shape:       Shape type string ("sphere", "cylinder", "box", "cone",
                     "ellipsoid", "torus").
        dimensions:  Shape parameters dict. Keys depend on shape:
                       sphere:   {"radius": float}
                       cylinder: {"radius": float, "length": float}
                       box:      {"width": float, "depth": float, "height": float}
                       cone:     {"radius": float, "height": float}
                       ellipsoid:{"a": float, "b": float, "c": float}
                       torus:    {"major_radius": float, "minor_radius": float}
        material_key: Key into sigma_ground MATERIALS dict ("iron", "rubber", etc.)
        position:    Initial position (x, y, z) in meters. Default: origin.
        velocity:    Initial velocity (vx, vy, vz) in m/s. Default: at rest.
        temperature: Temperature in K. Drives all derived properties.
        angular_velocity: Initial spin (wx, wy, wz) in rad/s. Future use.
        is_static:   If True, object doesn't move (walls, floors). Default: False.
    """

    def __init__(self, name, shape, dimensions, material_key,
                 position=(0, 0, 0), velocity=(0, 0, 0),
                 temperature=293.15, angular_velocity=(0, 0, 0),
                 is_static=False):
        self.name = name
        self.shape = shape
        self.dimensions = dict(dimensions)
        self.material_key = material_key
        self.position = tuple(position)
        self.velocity = tuple(velocity)
        self.temperature = float(temperature)
        self.angular_velocity = tuple(angular_velocity)
        self.is_static = is_static

    def __repr__(self):
        return (f"SimObject({self.name!r}, {self.shape}, "
                f"material={self.material_key!r})")


class SimEvent:
    """A named moment during simulation — triggers snapshot capture.

    Args:
        name:        Event label ("impact", "peak_height").
        condition:   Trigger type:
                       "collision" — any collision occurs
                       "collision:obj1,obj2" — specific pair collides
                       "t=<float>" — simulation reaches time t
                       "max_z:obj" — object reaches maximum height
                       "min_z:obj" — object reaches minimum height
                       "rest" — all objects have velocity < threshold
        description: Human-readable description of what this event means.
    """

    def __init__(self, name, condition, description=''):
        self.name = name
        self.condition = condition
        self.description = description

    def __repr__(self):
        return f"SimEvent({self.name!r}, {self.condition!r})"


class SimulationScene:
    """A complete simulation scenario — declarative, data-only.

    Args:
        name:              Short name ("free_fall", "baseball_bat").
        description:       What this simulation demonstrates.
        objects:           List of SimObject instances.
        environment:       Environment instance (from environment.py).
        events:            List of SimEvent triggers for snapshot capture.
        duration:          Total simulation time in seconds.
        snapshot_interval: Time between regular snapshots in seconds.
    """

    def __init__(self, name, description, objects, environment=None,
                 events=None, duration=1.0, snapshot_interval=0.01):
        self.name = name
        self.description = description
        self.objects = list(objects)
        # Lazy import to avoid circular dependency at module level
        if environment is None:
            from .environment import Environment
            environment = Environment()
        self.environment = environment
        self.events = list(events) if events else []
        self.duration = float(duration)
        self.snapshot_interval = float(snapshot_interval)

    def __repr__(self):
        return (f"SimulationScene({self.name!r}, "
                f"{len(self.objects)} objects, "
                f"duration={self.duration}s)")
