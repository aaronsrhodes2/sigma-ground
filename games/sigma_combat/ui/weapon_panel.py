"""
Weapon designer panel — left side of screen.

Type cards at top, then context-sensitive sliders below.
"""
import math
import pygame
from .widgets import Slider, Button, Dropdown, font

WEAPON_TYPES = [
    ('kinetic',      'KINETIC',      (200, 180, 80),  'Mass + velocity penetrator'),
    ('laser',        'LASER',        (80, 200, 255),  'Focused energy beam'),
    ('plasma',       'PLASMA',       (255, 130, 50),  'Superheated ionized gas'),
    ('explosive',    'EXPLOSIVE',    (255, 60, 60),   'Blast wave + thermal'),
    ('particle',     'PARTICLE',     (130, 255, 130), 'Relativistic beam'),
    ('gravitational','EXOTIC',       (180, 100, 255), 'Tidal/sigma-field disruption'),
]

# Material options for kinetic projectile
PROJ_MATERIALS = [
    ('steel_mild',      'Steel'),
    ('tungsten',        'Tungsten'),
    ('depleted_uranium','DU'),
    ('titanium',        'Titanium'),
    ('lead',            'Lead'),
    ('iron',            'Iron'),
]

PARTICLE_TYPES = [
    ('proton',   'Proton'),
    ('neutron',  'Neutron'),
    ('electron', 'Electron'),
]

C_LIGHT = 299_792_458.0


def _eng(v):
    if v == 0:
        return '0'
    exp = int(math.floor(math.log10(abs(v))))
    man = v / 10 ** exp
    if abs(man - round(man)) < 0.01:
        return f'{round(man):.0f}e{exp}'
    return f'{man:.2f}e{exp}'


def _vel_label(v):
    if v < 1:    return f'{v:.2f} m/s'
    if v < 1000: return f'{v:.0f} m/s'
    if v < C_LIGHT * 0.01: return f'{v/1000:.1f} km/s'
    return f'{v/C_LIGHT*100:.2f}% c'


def _mass_label(m):
    if m < 0.001: return f'{m*1000:.1f} g'
    if m < 1:     return f'{m*1000:.0f} g'
    if m < 1000:  return f'{m:.1f} kg'
    return f'{m/1000:.1f} t'


def _power_label(p):
    if p < 1:      return f'{p*1000:.0f} mW'
    if p < 1000:   return f'{p:.0f} W'
    if p < 1e6:    return f'{p/1000:.1f} kW'
    if p < 1e9:    return f'{p/1e6:.1f} MW'
    if p < 1e12:   return f'{p/1e9:.1f} GW'
    return f'{p/1e12:.2f} TW'


def _wave_label(nm):
    if nm > 1000:  return f'{nm/1000:.1f} um (IR)'
    if nm > 700:   return f'{nm:.0f} nm (near-IR)'
    if nm > 380:   return f'{nm:.0f} nm (visible)'
    if nm > 10:    return f'{nm:.0f} nm (UV)'
    if nm > 0.01:  return f'{nm:.3f} nm (X-ray)'
    return f'{nm:.4f} nm (gamma)'


def _temp_ev_label(ev):
    K = ev * 11604
    if K < 10000:   return f'{K:.0f} K'
    if K < 1e6:     return f'{K/1000:.0f} kK'
    if K < 1e9:     return f'{K/1e6:.1f} MK'
    return f'{ev:.0f} eV'


def _tnt_label(kg):
    if kg < 0.001: return f'{kg*1e6:.0f} ug TNT'
    if kg < 1:     return f'{kg*1000:.0f} g TNT'
    if kg < 1000:  return f'{kg:.1f} kg TNT'
    if kg < 1e6:   return f'{kg/1000:.0f} t TNT'
    return f'{kg/1e6:.0f} Mt TNT'


def _mev_label(e):
    if e < 1:      return f'{e*1000:.0f} keV'
    if e < 1000:   return f'{e:.0f} MeV'
    if e < 1e6:    return f'{e/1000:.0f} GeV'
    return f'{e/1e6:.2f} TeV'


class WeaponPanel:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.selected_type = 0  # index into WEAPON_TYPES
        self._build_type_buttons()
        self._build_sliders()

    def _card_rect(self, i):
        cols = 3
        cw = (self.rect.width - 10) // cols
        ch = 36
        r = i // cols
        c = i % cols
        return pygame.Rect(self.rect.left + 5 + c * cw,
                           self.rect.top + 8 + r * ch,
                           cw - 4, ch - 4)

    def _build_type_buttons(self):
        self.type_btns = []
        for i, (key, label, color, desc) in enumerate(WEAPON_TYPES):
            rect = self._card_rect(i)
            btn = Button(rect, label, color=(30, 40, 60),
                         text_color=color,
                         active=(i == self.selected_type),
                         active_color=tuple(min(255, c // 2) for c in color))
            self.type_btns.append(btn)

    def _slider_rect(self, idx):
        top = self.rect.top + 90 + idx * 52
        return pygame.Rect(self.rect.left + 10, top,
                           self.rect.width - 20, 20)

    def _build_sliders(self):
        x = self.rect.left + 10
        w = self.rect.width - 20
        base_y = self.rect.top + 90

        def sr(idx):
            return (x, base_y + idx * 52, w, 20)

        # ── Kinetic ──────────────────────────────────────────────
        self.k_mass = Slider(sr(0), 0.001, 5000, 0.01, 'Mass', log=True,
            format_fn=_mass_label,
            ticks=[(0.01,'bullet'),(10,'shell'),(1000,'tank')])
        self.k_vel = Slider(sr(1), 1, C_LIGHT * 0.95, 900, 'Velocity', log=True,
            format_fn=_vel_label,
            ticks=[(40,'throw'),(900,'rifle'),(3000,'rail'),(C_LIGHT*0.1,'0.1c')])
        self.k_radius = Slider(sr(2), 0.001, 0.5, 0.004, 'Radius', log=True,
            format_fn=lambda v: f'{v*1000:.1f} mm',
            ticks=[(0.004,'bullet'),(0.05,'cannon'),(0.2,'boulder')])
        self.k_mat = Dropdown((x, base_y + 3*52 + 18, w, 24),
                              PROJ_MATERIALS, selected=1, label='Penetrator material')

        # ── Laser ────────────────────────────────────────────────
        self.l_power = Slider(sr(0), 0.001, 1e12, 1e6, 'Power', log=True,
            format_fn=_power_label,
            ticks=[(0.005,'pointer'),(1000,'cutter'),(1e6,'military'),(1e9,'mega')])
        self.l_wave = Slider(sr(1), 0.001, 2000, 1064, 'Wavelength', log=True,
            format_fn=_wave_label,
            ticks=[(10,'X-ray'),(355,'UV'),(532,'green'),(1064,'IR')])
        self.l_dur = Slider(sr(2), 1e-12, 10, 1.0, 'Pulse duration', log=True,
            format_fn=lambda v: f'{v:.2e} s',
            ticks=[(1e-9,'ns'),(1e-3,'ms'),(1,'1s')])
        self.l_spot = Slider(sr(3), 0.0001, 0.5, 0.01, 'Spot radius', log=True,
            format_fn=lambda v: f'{v*1000:.1f} mm',
            ticks=[(0.001,'1mm'),(0.01,'1cm'),(0.1,'10cm')])

        # ── Plasma ───────────────────────────────────────────────
        self.p_temp = Slider(sr(0), 0.03, 10000, 10, 'Temperature', log=True,
            format_fn=_temp_ev_label,
            ticks=[(0.3,'candle'),(3,'arc'),(100,'edge'),(1000,'core')])
        self.p_pres = Slider(sr(1), 100, 1e9, 1e5, 'Pressure', log=True,
            format_fn=lambda v: f'{v:.2e} Pa',
            ticks=[(1e3,'low'),(1e5,'atm'),(1e7,'high'),(1e9,'confinement')])
        self.p_dur = Slider(sr(2), 0.001, 10, 0.1, 'Duration', log=False,
            format_fn=lambda v: f'{v:.2f} s',
            ticks=[(0.1,'flash'),(1,'pulse'),(5,'sustained')])
        self.p_rad = Slider(sr(3), 0.005, 1.0, 0.05, 'Contact radius', log=False,
            format_fn=lambda v: f'{v*100:.0f} cm',
            ticks=[(0.05,'5cm'),(0.2,'20cm'),(0.5,'50cm')])

        # ── Explosive ────────────────────────────────────────────
        self.e_yield = Slider(sr(0), 0.0001, 100000, 1, 'Explosive yield', log=True,
            format_fn=_tnt_label,
            ticks=[(0.15,'grenade'),(1,'bomb'),(1000,'airstrike'),(100000,'bunker')])
        self.e_standoff = Slider(sr(1), 0, 500, 0, 'Standoff distance', log=False,
            format_fn=lambda v: f'{v:.0f} m',
            ticks=[(0,'contact'),(10,'10m'),(100,'100m')])

        # ── Particle ─────────────────────────────────────────────
        self.pt_energy = Slider(sr(0), 0.1, 1e9, 100, 'Beam energy', log=True,
            format_fn=_mev_label,
            ticks=[(0.1,'100keV'),(100,'100MeV'),(1e6,'1TeV')])
        self.pt_current = Slider(sr(1), 1e-12, 0.01, 1e-6, 'Beam current', log=True,
            format_fn=lambda v: f'{v*1e9:.2g} nA' if v < 1e-6 else f'{v*1e6:.2g} uA',
            ticks=[(1e-9,'1nA'),(1e-6,'1uA'),(1e-3,'1mA')])
        self.pt_dur = Slider(sr(2), 0.001, 10, 0.5, 'Duration', log=False,
            format_fn=lambda v: f'{v:.2f} s')
        self.pt_mat = Dropdown((x, base_y + 3*52 + 18, w, 24),
                               PARTICLE_TYPES, selected=0, label='Particle type')

        # ── Gravitational ────────────────────────────────────────
        from sigma_ground.field.constants import SIGMA_CONV
        self.g_tidal = Slider(sr(0), 0, 1e13, 9.8, 'Tidal gradient', log=False,
            format_fn=lambda v: f'{v:.3g} m/s2/m',
            ticks=[(9.8,'Earth'),(1e6,'white dwarf'),(1e12,'neutron star')])
        self.g_sigma = Slider(sr(1), 0, SIGMA_CONV * 0.99, 0, 'sigma-field spike', log=False,
            format_fn=lambda v: f'{v:.3f} (max={SIGMA_CONV:.3f})',
            ticks=[(0,'zero'),(SIGMA_CONV*0.5,'half'),(SIGMA_CONV*0.95,'near-crit')])

    def _active_type(self):
        return WEAPON_TYPES[self.selected_type][0]

    def get_weapon(self) -> dict:
        wtype = self._active_type()
        base = {'type': wtype}

        if wtype == 'kinetic':
            base.update({
                'mass_kg': self.k_mass.value,
                'velocity_ms': self.k_vel.value,
                'radius_m': self.k_radius.value,
                'material': self.k_mat.value,
            })
        elif wtype == 'laser':
            base.update({
                'power_w': self.l_power.value,
                'wavelength_nm': self.l_wave.value,
                'pulse_duration_s': self.l_dur.value,
                'spot_radius_m': self.l_spot.value,
            })
        elif wtype == 'plasma':
            base.update({
                'temperature_ev': self.p_temp.value,
                'pressure_pa': self.p_pres.value,
                'duration_s': self.p_dur.value,
                'contact_radius_m': self.p_rad.value,
            })
        elif wtype == 'explosive':
            base.update({
                'tnt_kg': self.e_yield.value,
                'standoff_m': self.e_standoff.value,
            })
        elif wtype == 'particle':
            base.update({
                'energy_mev': self.pt_energy.value,
                'current_a': self.pt_current.value,
                'duration_s': self.pt_dur.value,
                'particle': self.pt_mat.value,
                'beam_radius_m': 0.005,
            })
        elif wtype == 'gravitational':
            base.update({
                'tidal_gradient_ms2': self.g_tidal.value,
                'sigma_spike': self.g_sigma.value,
            })

        return base

    def handle_event(self, event):
        # Type card selection
        for i, btn in enumerate(self.type_btns):
            if btn.handle_event(event):
                self.selected_type = i
                for j, b in enumerate(self.type_btns):
                    b.active = (j == i)
                return True

        # Active sliders
        for slider in self._active_sliders():
            if slider.handle_event(event):
                return True

        # Dropdowns
        dd = self._active_dropdown()
        if dd and dd.handle_event(event):
            return True

        return False

    def _active_sliders(self):
        wtype = self._active_type()
        mapping = {
            'kinetic':      [self.k_mass, self.k_vel, self.k_radius],
            'laser':        [self.l_power, self.l_wave, self.l_dur, self.l_spot],
            'plasma':       [self.p_temp, self.p_pres, self.p_dur, self.p_rad],
            'explosive':    [self.e_yield, self.e_standoff],
            'particle':     [self.pt_energy, self.pt_current, self.pt_dur],
            'gravitational':[self.g_tidal, self.g_sigma],
        }
        return mapping.get(wtype, [])

    def _active_dropdown(self):
        wtype = self._active_type()
        if wtype == 'kinetic':
            return self.k_mat
        if wtype == 'particle':
            return self.pt_mat
        return None

    def draw(self, surface):
        # Panel background
        pygame.draw.rect(surface, (20, 26, 38), self.rect)
        pygame.draw.rect(surface, (50, 65, 90), self.rect, 1)

        # Title
        title = font(14).render('WEAPON DESIGNER', True, (140, 160, 200))
        surface.blit(title, (self.rect.left + 6, self.rect.top - 20))

        # Type cards
        for i, (key, label, color, desc) in enumerate(WEAPON_TYPES):
            btn = self.type_btns[i]
            btn.active = (i == self.selected_type)
            btn.draw(surface)

        # Description of selected type
        wkey, wlbl, wcol, wdesc = WEAPON_TYPES[self.selected_type]
        d = font(11).render(wdesc, True, (100, 120, 145))
        surface.blit(d, (self.rect.left + 6, self.rect.top + 82))

        # Active sliders
        for slider in self._active_sliders():
            slider.draw(surface)

        # Active dropdown
        dd = self._active_dropdown()
        if dd:
            dd.draw(surface)
