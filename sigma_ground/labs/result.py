"""
Simulation results — structured output from a completed simulation.

All data is plain Python (lists, tuples, dicts, floats) — no numpy,
no external dependencies. Serializable to JSON for the MCP tool.
"""


class ObjectState:
    """State of one object at a moment in time.

    Args:
        name:           Object label.
        position:       (x, y, z) in meters.
        velocity:       (vx, vy, vz) in m/s.
        speed:          Scalar speed in m/s.
        kinetic_energy: Translational KE in joules.
        temperature:    Object temperature in K.
    """

    def __init__(self, name, position, velocity, speed,
                 kinetic_energy, temperature=293.15):
        self.name = name
        self.position = tuple(position)
        self.velocity = tuple(velocity)
        self.speed = float(speed)
        self.kinetic_energy = float(kinetic_energy)
        self.temperature = float(temperature)

    def to_dict(self):
        return {
            'name': self.name,
            'position': list(self.position),
            'velocity': list(self.velocity),
            'speed': self.speed,
            'kinetic_energy': self.kinetic_energy,
            'temperature': self.temperature,
        }


class CollisionEvent:
    """A collision that occurred during simulation.

    Args:
        time:              Simulation time of collision (s).
        object_a:          Name of first object.
        object_b:          Name of second object (or "ground").
        relative_velocity: Approach speed at contact (m/s).
        impulse:           Impulse magnitude (N*s).
        energy_dissipated: Energy lost to deformation (J).
    """

    def __init__(self, time, object_a, object_b,
                 relative_velocity=0.0, impulse=0.0,
                 energy_dissipated=0.0):
        self.time = float(time)
        self.object_a = object_a
        self.object_b = object_b
        self.relative_velocity = float(relative_velocity)
        self.impulse = float(impulse)
        self.energy_dissipated = float(energy_dissipated)

    def to_dict(self):
        return {
            'time': self.time,
            'objects': [self.object_a, self.object_b],
            'relative_velocity': self.relative_velocity,
            'impulse': self.impulse,
            'energy_dissipated': self.energy_dissipated,
        }


class Snapshot:
    """State of the simulation at a single moment.

    Args:
        time:                Simulation time (s).
        event_name:          Name of triggered event, or None for regular snapshot.
        objects:             List of ObjectState instances.
        total_kinetic_energy: Sum of all KE (J).
        total_potential_energy: Gravitational PE relative to ground (J).
        collisions_this_step: Collisions detected at this snapshot.
    """

    def __init__(self, time, objects, event_name=None,
                 total_kinetic_energy=0.0, total_potential_energy=0.0,
                 collisions_this_step=None):
        self.time = float(time)
        self.event_name = event_name
        self.objects = list(objects)
        self.total_kinetic_energy = float(total_kinetic_energy)
        self.total_potential_energy = float(total_potential_energy)
        self.collisions_this_step = list(collisions_this_step or [])

    def to_dict(self):
        return {
            'time': self.time,
            'event': self.event_name,
            'objects': [o.to_dict() for o in self.objects],
            'total_kinetic_energy': self.total_kinetic_energy,
            'total_potential_energy': self.total_potential_energy,
            'collisions': [c.to_dict() for c in self.collisions_this_step],
        }


class SimulationResult:
    """Complete output of a simulation run.

    Args:
        scene_name:    Name of the scene that was simulated.
        description:   Scene description.
        snapshots:     List of Snapshot instances (time-ordered).
        collisions:    All CollisionEvent instances from the run.
        duration:      Actual simulated time (s).
        steps_taken:   Number of integration steps.
        material_properties: Dict of cascaded properties per object.
        summary:       Human-readable summary dict.
    """

    def __init__(self, scene_name, description, snapshots, collisions,
                 duration, steps_taken, material_properties=None,
                 summary=None):
        self.scene_name = scene_name
        self.description = description
        self.snapshots = list(snapshots)
        self.collisions = list(collisions)
        self.duration = float(duration)
        self.steps_taken = int(steps_taken)
        self.material_properties = material_properties or {}
        self.summary = summary or {}

    @property
    def initial(self):
        """First snapshot."""
        return self.snapshots[0] if self.snapshots else None

    @property
    def final(self):
        """Last snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def to_dict(self):
        return {
            'scene_name': self.scene_name,
            'description': self.description,
            'duration': self.duration,
            'steps_taken': self.steps_taken,
            'num_snapshots': len(self.snapshots),
            'num_collisions': len(self.collisions),
            'material_properties': self.material_properties,
            'summary': self.summary,
            'initial_state': self.initial.to_dict() if self.initial else None,
            'final_state': self.final.to_dict() if self.final else None,
            'collisions': [c.to_dict() for c in self.collisions],
            'snapshots': [s.to_dict() for s in self.snapshots],
        }
