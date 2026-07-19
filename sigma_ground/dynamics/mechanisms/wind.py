"""Wind on a bladed rotor — a NATURAL drive (the preferred mode of the
force-provenance doctrine: nothing plugged, the machine turns because
simulated air pushes on real blade area).

Model: FLAT-PLATE NORMAL FORCE per blade —
    u   = wind - v_blade_point          (relative flow at the blade centroid)
    u_n = u . n_hat                     (component normal to the blade face)
    F   = 1/2 * rho_air * C_N * A * |u_n| * u_n * n_hat    (at the centroid)
    tau = r x F                         (about the body CoM)

SIMPLIFIED_MODEL, flagged plainly: this is the classic flat-plate normal-
force law (C_N ~ 1.28, the standard flat-plate drag coefficient for a
plate normal to flow — same constant family as labs.forces' Reynolds-regime
C_d table), evaluated once at each blade's centroid. It does NOT model
lift/drag decomposition, spanwise variation (blade-element momentum
theory), wake losses, or blade-blade interference. What it DOES give,
honestly and emergently: torque appears only when blades are PITCHED
(a face-on flat blade yields zero moment about the axis — geometry, not
scripting), the rotor spins up monotonically, and it self-limits at the
tip-speed ratio where the blade's own tangential motion cancels the
normal inflow:

    u_n = U*cos(beta) - omega*R_c*sin(beta) = 0
    =>  omega_terminal = U * cot(beta) / R_c        (closed form, gated)

rho_air default 1.225 kg/m^3 (sea-level standard atmosphere).
"""
import math

from ..vec import Vec3
from ..quat import qrot

RHO_AIR_SEA_LEVEL = 1.225      # kg/m^3, standard atmosphere
C_N_FLAT_PLATE = 1.28          # flat-plate normal-force coefficient (flagged)


class Blade:
    """One blade: centroid + face normal in the ROTOR BODY frame, + area."""

    def __init__(self, centroid_local, normal_local, area_m2):
        self.centroid_local = centroid_local
        n = normal_local
        ln = math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
        self.normal_local = Vec3(n.x / ln, n.y / ln, n.z / ln)
        self.area_m2 = float(area_m2)


class RotorWind:
    """External-forces callback (the stepper's (force, torque) contract)
    applying flat-plate wind forces to ONE rotor parcel's blades; every
    other parcel gets zero (compose multiple instances for multiple rotors).
    """

    def __init__(self, parcel, blades, wind_velocity,
                 rho_air=RHO_AIR_SEA_LEVEL, c_n=C_N_FLAT_PLATE):
        self.parcel = parcel
        self.blades = list(blades)
        self.wind = wind_velocity
        self.rho_air = float(rho_air)
        self.c_n = float(c_n)

    def __call__(self, p):
        if p is not self.parcel:
            return Vec3(0.0, 0.0, 0.0)
        F_tot = Vec3(0.0, 0.0, 0.0)
        tau_tot = Vec3(0.0, 0.0, 0.0)
        q = p.orientation
        w = p.angular_velocity
        for b in self.blades:
            c = b.centroid_local
            r_t = qrot(q, (c.x, c.y, c.z))
            r = Vec3(r_t[0], r_t[1], r_t[2])            # world offset from CoM
            n_t = qrot(q, (b.normal_local.x, b.normal_local.y, b.normal_local.z))
            n = Vec3(n_t[0], n_t[1], n_t[2])            # world face normal
            v_pt = p.velocity + w.cross(r)              # blade-point velocity
            u = self.wind - v_pt                        # relative flow
            u_n = u.dot(n)
            mag = 0.5 * self.rho_air * self.c_n * b.area_m2 * abs(u_n) * u_n
            F = n * mag
            F_tot = F_tot + F
            tau_tot = tau_tot + r.cross(F)
        return F_tot, tau_tot


def build_rotor_blades(n_blades, radius_centroid_m, length_m, chord_m,
                       pitch_rad, axis="x"):
    """Blade list for a rotor whose spin axis is +x (wind convention:
    horizontal-axis machine facing the flow). Blade b sits at azimuth
    phi_b = 2*pi*b/n, radially outward in the yz-plane, its face normal
    pitched by ``pitch_rad`` about its own radial axis (zero pitch = face
    square to the wind = zero torque; the pitch is what turns axial flow
    into tangential force — real windmill geometry, not a plug).

    Returns (blades, per_blade_area). Geometry convention shared with the
    renderer's Rotated-Box leaves so physics and picture come from the
    same numbers."""
    if axis != "x":
        raise NotImplementedError("rotor axis convention is +x (facing the wind)")
    area = length_m * chord_m
    blades = []
    for b in range(n_blades):
        phi = 2.0 * math.pi * b / n_blades
        d = Vec3(0.0, math.cos(phi), math.sin(phi))        # radial
        t = Vec3(0.0, -math.sin(phi), math.cos(phi))       # tangential (x_hat x d)
        centroid = d * radius_centroid_m
        n = Vec3(math.cos(pitch_rad), 0.0, 0.0) + t * math.sin(pitch_rad)
        blades.append(Blade(centroid, n, area))
    return blades, area


def terminal_omega(wind_speed, pitch_rad, radius_centroid_m):
    """The self-limiting spin rate (closed form, see module docstring):
    where blade tangential motion cancels the normal inflow."""
    return wind_speed / (math.tan(pitch_rad) * radius_centroid_m)


__all__ = ["Blade", "RotorWind", "build_rotor_blades", "terminal_omega",
           "RHO_AIR_SEA_LEVEL", "C_N_FLAT_PLATE"]
