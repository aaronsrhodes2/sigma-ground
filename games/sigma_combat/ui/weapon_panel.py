"""
Weapon panel — preset-first design.

Primary UI: scrollable card list of known weapons grouped by era/type.
Custom editor: collapsible section at bottom with type-picker + sliders.
"""
import math
import pygame
from .widgets import Slider, Button, Dropdown, font
from ..presets import WEAPON_PRESETS, get_weapon as _get_preset

C_LIGHT = 299_792_458.0

WEAPON_TYPE_COLORS = {
    'kinetic':      (200, 180,  80),
    'laser':        ( 80, 200, 255),
    'plasma':       (255, 130,  50),
    'explosive':    (255,  60,  60),
    'particle':     (130, 255, 130),
    'gravitational':(180, 100, 255),
}

WEAPON_GROUPS = [
    ('ANCIENT',         ['default', 'throwing_spear', 'crossbow_bolt', 'cannonball']),
    ('FIREARMS',        ['pistol_9mm', 'ak47', 'm2_browning', 'armor_piercing_30mm']),
    ('ANTI-ARMOR',      ['rpg7', 'atgm', 'sabot_apfsds']),
    ('DIRECTED ENERGY', ['laser_designator', 'hel_100kw', 'hel_megawatt']),
    ('SCI-FI / EXOTIC', ['railgun_hypersonic', 'railgun_relativistic', 'plasma_cannon',
                         'plasma_torch_industrial', 'proton_cannon', 'neutron_bomb',
                         'gamma_laser', 'tidal_disruptor', 'sigma_spike_weapon',
                         'city_buster', 'phaser_stun', 'photon_torpedo']),
]

CARD_H  = 28
GROUP_H = 20

# Keep custom editor types list for compatibility
WEAPON_TYPES = [
    ('kinetic',      'KINETIC',      (200, 180, 80),  'Mass + velocity penetrator'),
    ('laser',        'LASER',        (80, 200, 255),  'Focused energy beam'),
    ('plasma',       'PLASMA',       (255, 130, 50),  'Superheated ionized gas'),
    ('explosive',    'EXPLOSIVE',    (255, 60, 60),   'Blast wave + thermal'),
    ('particle',     'PARTICLE',     (130, 255, 130), 'Relativistic beam'),
    ('gravitational','EXOTIC',       (180, 100, 255), 'Tidal/sigma-field disruption'),
]

PROJ_MATERIALS = [
    ('steel_mild','Steel'), ('tungsten','Tungsten'), ('depleted_uranium','DU'),
    ('titanium','Titanium'), ('lead','Lead'), ('iron','Iron'),
]
PARTICLE_TYPES = [('proton','Proton'), ('neutron','Neutron'), ('electron','Electron')]


# ── Label helpers ─────────────────────────────────────────────────────────────

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
    if p < 1000:  return f'{p:.0f} W'
    if p < 1e6:   return f'{p/1000:.1f} kW'
    if p < 1e9:   return f'{p/1e6:.1f} MW'
    if p < 1e12:  return f'{p/1e9:.1f} GW'
    return f'{p/1e12:.2f} TW'

def _wave_label(nm):
    if nm > 700:  return f'{nm:.0f} nm (IR/vis)'
    if nm > 380:  return f'{nm:.0f} nm (vis)'
    if nm > 10:   return f'{nm:.0f} nm (UV)'
    if nm > 0.01: return f'{nm:.3f} nm (X-ray)'
    return f'{nm:.4f} nm (gamma)'

def _temp_ev_label(ev):
    K = ev * 11604
    if K < 10000:  return f'{K:.0f} K'
    if K < 1e6:    return f'{K/1000:.0f} kK'
    return f'{K/1e6:.1f} MK'

def _tnt_label(kg):
    if kg < 1:     return f'{kg*1000:.0f} g TNT'
    if kg < 1000:  return f'{kg:.1f} kg TNT'
    if kg < 1e6:   return f'{kg/1000:.0f} t TNT'
    return f'{kg/1e6:.0f} Mt TNT'

def _mev_label(e):
    if e < 1:    return f'{e*1000:.0f} keV'
    if e < 1000: return f'{e:.0f} MeV'
    if e < 1e6:  return f'{e/1000:.0f} GeV'
    return f'{e/1e6:.2f} TeV'


# ── WeaponPanel ───────────────────────────────────────────────────────────────

class WeaponPanel:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self._selected_preset = 'default'
        self._custom_open     = False
        self._scroll_y        = 0
        self._max_scroll      = 0
        self.selected_type    = 0
        self._build_type_buttons()
        self._build_sliders()

    # ── Preset helpers ────────────────────────────────────────────────────────

    def _active_type(self):
        return WEAPON_TYPES[self.selected_type][0]

    def get_weapon(self) -> dict:
        if not self._custom_open:
            return _get_preset(self._selected_preset)
        return self._sliders_to_weapon()

    def _sliders_to_weapon(self):
        wtype = self._active_type()
        base  = {'type': wtype}
        if wtype == 'kinetic':
            base.update({'mass_kg': self.k_mass.value, 'velocity_ms': self.k_vel.value,
                         'radius_m': self.k_radius.value, 'material': self.k_mat.value})
        elif wtype == 'laser':
            base.update({'power_w': self.l_power.value, 'wavelength_nm': self.l_wave.value,
                         'pulse_duration_s': self.l_dur.value, 'spot_radius_m': self.l_spot.value})
        elif wtype == 'plasma':
            base.update({'temperature_ev': self.p_temp.value, 'pressure_pa': self.p_pres.value,
                         'duration_s': self.p_dur.value, 'contact_radius_m': self.p_rad.value})
        elif wtype == 'explosive':
            base.update({'tnt_kg': self.e_yield.value, 'standoff_m': self.e_standoff.value})
        elif wtype == 'particle':
            base.update({'energy_mev': self.pt_energy.value, 'current_a': self.pt_current.value,
                         'duration_s': self.pt_dur.value, 'particle': self.pt_mat.value,
                         'beam_radius_m': 0.005})
        elif wtype == 'gravitational':
            base.update({'tidal_gradient_ms2': self.g_tidal.value,
                         'sigma_spike': self.g_sigma.value})
        return base

    # ── Widget builders ───────────────────────────────────────────────────────

    def _card_rect(self, i):
        cols, cw, ch = 3, (self.rect.width - 10) // 3, 36
        return pygame.Rect(self.rect.left + 5 + (i % cols) * cw,
                           self.rect.top + 8 + (i // cols) * ch,
                           cw - 4, ch - 4)

    def _build_type_buttons(self):
        self.type_btns = []
        for i, (key, label, color, _desc) in enumerate(WEAPON_TYPES):
            btn = Button(self._card_rect(i), label, color=(30, 40, 60),
                         text_color=color, active=(i == self.selected_type),
                         active_color=tuple(min(255, c // 2) for c in color))
            self.type_btns.append(btn)

    def _build_sliders(self):
        x = self.rect.left + 10
        w = self.rect.width - 20
        by = self.rect.top + 90

        def sr(i): return (x, by + i * 52, w, 20)

        self.k_mass   = Slider(sr(0), 0.001, 5000, 0.01, 'Mass', log=True,
            format_fn=_mass_label, ticks=[(0.01,'bullet'),(10,'shell'),(1000,'tank')])
        self.k_vel    = Slider(sr(1), 1, C_LIGHT*0.95, 900, 'Velocity', log=True,
            format_fn=_vel_label, ticks=[(40,'throw'),(900,'rifle'),(3000,'rail'),(C_LIGHT*0.1,'0.1c')])
        self.k_radius = Slider(sr(2), 0.001, 0.5, 0.004, 'Radius', log=True,
            format_fn=lambda v: f'{v*1000:.1f} mm',
            ticks=[(0.004,'bullet'),(0.05,'cannon'),(0.2,'boulder')])
        self.k_mat    = Dropdown((x, by+3*52+18, w, 24), PROJ_MATERIALS, selected=1,
                                 label='Penetrator material')

        self.l_power = Slider(sr(0), 0.001, 1e12, 1e6, 'Power', log=True,
            format_fn=_power_label, ticks=[(0.005,'ptr'),(1000,'cut'),(1e6,'mil'),(1e9,'mega')])
        self.l_wave  = Slider(sr(1), 0.001, 2000, 1064, 'Wavelength', log=True,
            format_fn=_wave_label, ticks=[(10,'X-ray'),(355,'UV'),(532,'grn'),(1064,'IR')])
        self.l_dur   = Slider(sr(2), 1e-12, 10, 1.0, 'Pulse duration', log=True,
            format_fn=lambda v: f'{v:.2e} s', ticks=[(1e-9,'ns'),(1e-3,'ms'),(1,'1s')])
        self.l_spot  = Slider(sr(3), 0.0001, 0.5, 0.01, 'Spot radius', log=True,
            format_fn=lambda v: f'{v*1000:.1f} mm',
            ticks=[(0.001,'1mm'),(0.01,'1cm'),(0.1,'10cm')])

        self.p_temp = Slider(sr(0), 0.03, 10000, 10, 'Temperature', log=True,
            format_fn=_temp_ev_label, ticks=[(0.3,'candle'),(3,'arc'),(100,'edge'),(1000,'core')])
        self.p_pres = Slider(sr(1), 100, 1e9, 1e5, 'Pressure', log=True,
            format_fn=lambda v: f'{v:.2e} Pa',
            ticks=[(1e3,'low'),(1e5,'atm'),(1e7,'high')])
        self.p_dur  = Slider(sr(2), 0.001, 10, 0.1, 'Duration', log=False,
            format_fn=lambda v: f'{v:.2f} s', ticks=[(0.1,'flash'),(5,'sustained')])
        self.p_rad  = Slider(sr(3), 0.005, 1.0, 0.05, 'Contact radius', log=False,
            format_fn=lambda v: f'{v*100:.0f} cm', ticks=[(0.05,'5cm'),(0.5,'50cm')])

        self.e_yield    = Slider(sr(0), 0.0001, 100000, 1, 'Explosive yield', log=True,
            format_fn=_tnt_label,
            ticks=[(0.15,'grenade'),(1,'bomb'),(1000,'strike'),(100000,'bunker')])
        self.e_standoff = Slider(sr(1), 0, 500, 0, 'Standoff distance', log=False,
            format_fn=lambda v: f'{v:.0f} m', ticks=[(0,'contact'),(10,'10m'),(100,'100m')])

        self.pt_energy  = Slider(sr(0), 0.1, 1e9, 100, 'Beam energy', log=True,
            format_fn=_mev_label, ticks=[(0.1,'100keV'),(100,'100MeV'),(1e6,'1TeV')])
        self.pt_current = Slider(sr(1), 1e-12, 0.01, 1e-6, 'Beam current', log=True,
            format_fn=lambda v: f'{v*1e9:.2g} nA' if v < 1e-6 else f'{v*1e6:.2g} uA',
            ticks=[(1e-9,'1nA'),(1e-6,'1uA'),(1e-3,'1mA')])
        self.pt_dur     = Slider(sr(2), 0.001, 10, 0.5, 'Duration', log=False,
            format_fn=lambda v: f'{v:.2f} s')
        self.pt_mat     = Dropdown((x, by+3*52+18, w, 24), PARTICLE_TYPES, selected=0,
                                   label='Particle type')

        from sigma_ground.field.constants import SIGMA_CONV
        self.g_tidal = Slider(sr(0), 0, 1e13, 9.8, 'Tidal gradient', log=False,
            format_fn=lambda v: f'{v:.3g} m/s2/m',
            ticks=[(9.8,'Earth'),(1e6,'w.dwarf'),(1e12,'n.star')])
        self.g_sigma = Slider(sr(1), 0, SIGMA_CONV*0.99, 0, 'Sigma-field spike', log=False,
            format_fn=lambda v: f'{v:.3f}',
            ticks=[(0,'zero'),(SIGMA_CONV*0.5,'half'),(SIGMA_CONV*0.95,'crit')])

    def _active_sliders(self):
        return {
            'kinetic':      [self.k_mass, self.k_vel, self.k_radius],
            'laser':        [self.l_power, self.l_wave, self.l_dur, self.l_spot],
            'plasma':       [self.p_temp, self.p_pres, self.p_dur, self.p_rad],
            'explosive':    [self.e_yield, self.e_standoff],
            'particle':     [self.pt_energy, self.pt_current, self.pt_dur],
            'gravitational':[self.g_tidal, self.g_sigma],
        }.get(self._active_type(), [])

    def _active_dropdown(self):
        wtype = self._active_type()
        if wtype == 'kinetic':  return self.k_mat
        if wtype == 'particle': return self.pt_mat
        return None

    # ── List geometry helpers ─────────────────────────────────────────────────

    def _list_area(self):
        return pygame.Rect(self.rect.left, self.rect.top,
                           self.rect.width, self.rect.height - 32)

    def _total_list_height(self):
        h = 0
        for group, keys in WEAPON_GROUPS:
            h += GROUP_H
            h += sum(1 for k in keys if k in WEAPON_PRESETS) * CARD_H
        return h

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, event):
        la = self._list_area()

        # Toggle custom editor
        toggle_r = pygame.Rect(self.rect.left + 4, self.rect.bottom - 28,
                               self.rect.width - 8, 24)
        if event.type == pygame.MOUSEBUTTONDOWN and toggle_r.collidepoint(event.pos):
            self._custom_open = not self._custom_open
            return True

        if not self._custom_open:
            # Scroll preset list
            if event.type == pygame.MOUSEWHEEL and la.collidepoint(pygame.mouse.get_pos()):
                self._scroll_y = max(0, min(self._max_scroll,
                                            self._scroll_y - event.y * 18))
                return True
            # Card click
            if event.type == pygame.MOUSEBUTTONDOWN and la.collidepoint(event.pos):
                y = la.top + 2 - self._scroll_y
                for group, keys in WEAPON_GROUPS:
                    y += GROUP_H
                    for key in keys:
                        if key not in WEAPON_PRESETS:
                            continue
                        card_r = pygame.Rect(la.left + 3, y, la.width - 6, CARD_H - 2)
                        if card_r.collidepoint(event.pos):
                            self._selected_preset = key
                            return True
                        y += CARD_H
        else:
            # Custom editor events
            for i, btn in enumerate(self.type_btns):
                if btn.handle_event(event):
                    self.selected_type = i
                    for j, b in enumerate(self.type_btns):
                        b.active = (j == i)
                    return True
            for sl in self._active_sliders():
                if sl.handle_event(event):
                    return True
            dd = self._active_dropdown()
            if dd and dd.handle_event(event):
                return True

        return False

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surface):
        pygame.draw.rect(surface, (20, 26, 38), self.rect)

        la = self._list_area()

        if not self._custom_open:
            self._draw_preset_list(surface, la)
        else:
            self._draw_custom_editor(surface)

        # Toggle button
        toggle_r = pygame.Rect(self.rect.left + 4, self.rect.bottom - 28,
                               self.rect.width - 8, 24)
        label = 'CUSTOM EDITOR  ▲' if self._custom_open else '▼  CUSTOM EDITOR'
        tc = (200, 80, 80) if self._custom_open else (80, 100, 140)
        bg = (35, 18, 18) if self._custom_open else (22, 26, 38)
        pygame.draw.rect(surface, bg, toggle_r, border_radius=4)
        pygame.draw.rect(surface, tc, toggle_r, 1, border_radius=4)
        tl = font(11).render(label, True, tc)
        surface.blit(tl, (toggle_r.centerx - tl.get_width() // 2,
                           toggle_r.centery - tl.get_height() // 2))

    def _draw_preset_list(self, surface, la):
        # Draw into a clipped surface to contain the scroll
        clip = pygame.Surface((la.width, la.height))
        clip.fill((20, 26, 38))

        y    = 2 - self._scroll_y
        total_h = 0

        for group, keys in WEAPON_GROUPS:
            # Group header
            gh = font(9).render(group, True, (70, 90, 120))
            clip.blit(gh, (5, y + 3))
            pygame.draw.line(clip, (40, 55, 75),
                             (5, y + GROUP_H - 3), (la.width - 5, y + GROUP_H - 3), 1)
            y        += GROUP_H
            total_h  += GROUP_H

            for key in keys:
                if key not in WEAPON_PRESETS:
                    continue
                preset   = WEAPON_PRESETS[key]
                wtype    = preset['weapon']['type']
                color    = WEAPON_TYPE_COLORS.get(wtype, (150, 150, 150))
                selected = (key == self._selected_preset)

                card_r = pygame.Rect(3, y, la.width - 6, CARD_H - 2)

                if selected:
                    pygame.draw.rect(clip, (55, 22, 22), card_r, border_radius=3)
                    pygame.draw.rect(clip, (200, 60, 60), card_r, 1, border_radius=3)
                else:
                    pygame.draw.rect(clip, (28, 34, 50), card_r, border_radius=3)

                # Type stripe
                pygame.draw.rect(clip, color,
                                 pygame.Rect(3, y, 4, CARD_H - 2), border_radius=2)

                # Name
                name_col = (255, 215, 215) if selected else (195, 210, 230)
                name_s = font(11).render(preset['name'][:32], True, name_col)
                clip.blit(name_s, (10, y + (CARD_H - 2 - name_s.get_height()) // 2))

                # Type badge (right side)
                badge = font(9).render(wtype[:4].upper(), True,
                                       color if not selected else (255, 180, 180))
                clip.blit(badge, (la.width - badge.get_width() - 6,
                                  y + (CARD_H - 2 - badge.get_height()) // 2))

                y       += CARD_H
                total_h += CARD_H

        self._max_scroll = max(0, total_h - la.height + 20)
        surface.blit(clip, la.topleft)

        # Thin scrollbar
        if self._max_scroll > 0:
            frac  = self._scroll_y / self._max_scroll
            sb_h  = max(24, int(la.height * la.height / (total_h + 10)))
            sb_y  = la.top + int(frac * (la.height - sb_h))
            pygame.draw.rect(surface, (60, 80, 110),
                             pygame.Rect(la.right - 4, sb_y, 3, sb_h), border_radius=2)

        # Selected preset name at bottom of list (just above toggle)
        if self._selected_preset in WEAPON_PRESETS:
            sel_name = WEAPON_PRESETS[self._selected_preset]['name']
            sn = font(10).render(f'Selected: {sel_name[:28]}', True, (160, 100, 100))
            surface.blit(sn, (self.rect.left + 6, self.rect.bottom - 44))

    def _draw_custom_editor(self, surface):
        # Type cards
        for i, (key, label, color, _) in enumerate(WEAPON_TYPES):
            btn = self.type_btns[i]
            btn.active = (i == self.selected_type)
            btn.draw(surface)

        # Selected type description
        _, _, wcol, wdesc = WEAPON_TYPES[self.selected_type]
        d = font(11).render(wdesc, True, (100, 120, 145))
        surface.blit(d, (self.rect.left + 6, self.rect.top + 82))

        # Sliders
        for sl in self._active_sliders():
            sl.draw(surface)

        dd = self._active_dropdown()
        if dd:
            dd.draw(surface)
