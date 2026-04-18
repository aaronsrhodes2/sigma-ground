"""
sigma-combat — Weapons vs. Armor

All damage is derived from sigma-ground material physics.
No HP. No tuning knobs. Just thermodynamics, yield stress, and bond failure.

Run:  python games/sigma_combat/main.py
"""
import sys
import os

# Ensure sigma-ground is on the path when running from the repo root
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import pygame

from .physics.solver import solve
from .render.cross_section import render as render_cross_section
from .render.results_overlay import render as render_overlay
from .render import sounds
from .ui.weapon_panel import WeaponPanel
from .ui.armor_panel import ArmorPanel
from .ui.env_panel import EnvPanel
from .presets import get_armor, get_weapon, armor_menu, weapon_menu, WEAPON_PRESETS

# ── Layout ──────────────────────────────────────────────────────
SCREEN_W = 1340
SCREEN_H = 800

TOP_H = 58       # environment bar
BOTTOM_H = 90    # fire button + results bar
MAIN_H = SCREEN_H - TOP_H - BOTTOM_H

LEFT_W = 330     # weapon panel
ARMOR_W = 260    # armor builder
XS_W = SCREEN_W - LEFT_W - ARMOR_W  # cross-section (750px)

FPS = 60
TITLE = 'sigma-combat — Weapons vs. Armor'

# Colors
BG = (12, 16, 24)
PANEL_BG = (18, 24, 36)
DIVIDER = (45, 60, 85)

# Fire button
FIRE_COLOR = (200, 30, 30)
FIRE_HOVER = (240, 60, 60)
FIRE_ACTIVE = (255, 100, 100)


def main():
    pygame.init()
    pygame.display.set_caption(TITLE)
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    # Panel rectangles (with 5px padding inside each)
    env_rect    = pygame.Rect(0, 0, SCREEN_W, TOP_H)
    weapon_rect = pygame.Rect(5, TOP_H + 20, LEFT_W - 10, MAIN_H - 25)
    armor_rect  = pygame.Rect(LEFT_W + 5, TOP_H + 20, ARMOR_W - 10, MAIN_H - 25)
    xs_rect     = pygame.Rect(LEFT_W + ARMOR_W, TOP_H, XS_W, MAIN_H)
    bottom_rect = pygame.Rect(0, TOP_H + MAIN_H, SCREEN_W, BOTTOM_H)

    env_panel    = EnvPanel(env_rect)
    weapon_panel = WeaponPanel(weapon_rect)
    armor_panel  = ArmorPanel(armor_rect)

    # Load default preset: flint arrow vs 7-layer leather
    armor_panel.layers = get_armor('default')
    _default_w = get_weapon('default')
    weapon_panel.k_mass.value  = _default_w['mass_kg']
    weapon_panel.k_vel.value   = _default_w['velocity_ms']
    weapon_panel.k_radius.value = _default_w['radius_m']

    xs_surface = pygame.Surface((xs_rect.width, xs_rect.height))

    # Results state
    result = None
    last_weapon = None
    firing_flash = 0  # countdown for fire button flash

    # Preset dropdowns (bottom bar)
    from .ui.widgets import Dropdown as _DD
    _wmenu = weapon_menu()
    _amenu = armor_menu()
    weapon_preset_dd = _DD(
        (10, bottom_rect.top + 10, 320, 24),
        _wmenu, selected=0, label='')
    armor_preset_dd = _DD(
        (bottom_rect.right - 330, bottom_rect.top + 10, 320, 24),
        _amenu, selected=0, label='')

    # Fire button
    fire_rect = pygame.Rect(SCREEN_W // 2 - 120, bottom_rect.top + 10, 240, 50)
    fire_hovered = False

    def do_fire():
        nonlocal result, last_weapon, firing_flash
        weapon = weapon_panel.get_weapon()
        layers = armor_panel.get_layers()
        if not layers:
            return
        env = env_panel.environment
        dist = env_panel.dist_slider.value
        result = solve(weapon, layers, environment=env, distance_m=dist)
        last_weapon = weapon
        firing_flash = 8
        sounds.play_fire(weapon['type'], env)
        pygame.time.set_timer(pygame.USEREVENT + 1, 300, loops=1)  # delayed impact sound

    running = True
    while running:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    do_fire()
            elif event.type == pygame.USEREVENT + 1:
                if result:
                    sounds.play_impact(result, env_panel.environment)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if fire_rect.collidepoint(mx, my):
                    do_fire()
                    continue
            elif event.type == pygame.MOUSEMOTION:
                fire_hovered = fire_rect.collidepoint(event.pos)

            env_panel.handle_event(event)
            weapon_panel.handle_event(event)
            armor_panel.handle_event(event)

            # Preset dropdowns
            if weapon_preset_dd.handle_event(event):
                wkey = weapon_preset_dd.value
                w = get_weapon(wkey)
                wtype = w.get('type', 'kinetic')
                # Select weapon type in panel
                from .ui.weapon_panel import WEAPON_TYPES as _WT
                for i, (k, *_) in enumerate(_WT):
                    if k == wtype:
                        weapon_panel.selected_type = i
                        for j, b in enumerate(weapon_panel.type_btns):
                            b.active = (j == i)
                        break
                # Load params
                if wtype == 'kinetic':
                    weapon_panel.k_mass.value   = w.get('mass_kg', 0.01)
                    weapon_panel.k_vel.value    = w.get('velocity_ms', 900)
                    weapon_panel.k_radius.value = w.get('radius_m', 0.004)
                elif wtype == 'laser':
                    weapon_panel.l_power.value = w.get('power_w', 1e6)
                    weapon_panel.l_wave.value  = w.get('wavelength_nm', 1064)
                    weapon_panel.l_dur.value   = w.get('pulse_duration_s', 1.0)
                    weapon_panel.l_spot.value  = w.get('spot_radius_m', 0.01)
                elif wtype == 'plasma':
                    weapon_panel.p_temp.value = w.get('temperature_ev', 10)
                    weapon_panel.p_pres.value = w.get('pressure_pa', 1e5)
                    weapon_panel.p_dur.value  = w.get('duration_s', 0.1)
                    weapon_panel.p_rad.value  = w.get('contact_radius_m', 0.05)
                elif wtype == 'explosive':
                    weapon_panel.e_yield.value    = w.get('tnt_kg', 1)
                    weapon_panel.e_standoff.value = w.get('standoff_m', 0)
                elif wtype == 'particle':
                    weapon_panel.pt_energy.value  = w.get('energy_mev', 100)
                    weapon_panel.pt_current.value = w.get('current_a', 1e-6)
                    weapon_panel.pt_dur.value     = w.get('duration_s', 0.5)
                elif wtype == 'gravitational':
                    weapon_panel.g_tidal.value = w.get('tidal_gradient_ms2', 0)
                    weapon_panel.g_sigma.value = w.get('sigma_spike', 0)
                result = None

            if armor_preset_dd.handle_event(event):
                armor_panel.layers = get_armor(armor_preset_dd.value)
                result = None

        # ── Draw ────────────────────────────────────────────────
        screen.fill(BG)

        # Environment bar
        env_panel.draw(screen)

        # Panel labels (above panels)
        f14 = pygame.font.SysFont('consolas', 14)
        screen.blit(f14.render('WEAPON DESIGNER', True, (100, 130, 170)),
                    (weapon_rect.left, TOP_H + 2))
        screen.blit(f14.render('ARMOR LAYERS', True, (100, 130, 170)),
                    (armor_rect.left, TOP_H + 2))
        screen.blit(f14.render('CROSS-SECTION', True, (80, 110, 150)),
                    (xs_rect.left + 6, TOP_H + 2))

        # Dividers
        pygame.draw.line(screen, DIVIDER, (LEFT_W, TOP_H), (LEFT_W, TOP_H + MAIN_H))
        pygame.draw.line(screen, DIVIDER, (LEFT_W + ARMOR_W, TOP_H), (LEFT_W + ARMOR_W, TOP_H + MAIN_H))
        pygame.draw.line(screen, DIVIDER, (0, TOP_H + MAIN_H), (SCREEN_W, TOP_H + MAIN_H))

        # Panels
        weapon_panel.draw(screen)
        armor_panel.draw(screen)

        # Cross-section
        xs_surface.fill((14, 18, 28))
        wtype = weapon_panel._active_type()
        layers = armor_panel.get_layers()
        render_cross_section(xs_surface, layers, result=result, weapon_type=wtype)
        screen.blit(xs_surface, xs_rect.topleft)

        # Results overlay
        if result:
            overlay_rect = pygame.Rect(xs_rect.left + 4,
                                       xs_rect.bottom - 160,
                                       xs_rect.width - 8, 155)
            render_overlay(screen, result, overlay_rect)

        # Preset dropdowns
        f11 = pygame.font.SysFont('consolas', 11)
        screen.blit(f11.render('Weapon preset:', True, (70, 90, 110)),
                    (10, bottom_rect.top - 2))
        screen.blit(f11.render('Armor preset:', True, (70, 90, 110)),
                    (bottom_rect.right - 330, bottom_rect.top - 2))
        weapon_preset_dd.draw(screen)
        armor_preset_dd.draw(screen)

        # Fire button
        if firing_flash > 0:
            firing_flash -= 1
            fc = FIRE_ACTIVE
        elif fire_hovered:
            fc = FIRE_HOVER
        else:
            fc = FIRE_COLOR

        pygame.draw.rect(screen, fc, fire_rect, border_radius=8)
        pygame.draw.rect(screen, (255, 150, 150), fire_rect, 2, border_radius=8)
        f_lbl = pygame.font.SysFont('consolas', 22, bold=True).render('F I R E !', True, (255, 240, 240))
        screen.blit(f_lbl, (fire_rect.centerx - f_lbl.get_width() // 2,
                             fire_rect.centery - f_lbl.get_height() // 2))

        # SPACE hint
        hint = pygame.font.SysFont('consolas', 11).render('[SPACE] to fire', True, (60, 80, 100))
        screen.blit(hint, (fire_rect.centerx - hint.get_width() // 2,
                            fire_rect.bottom + 4))

        # Quick stats in bottom bar corners
        if result:
            verdict_col = (255, 80, 80) if result.breached else (80, 220, 100)
            vs = pygame.font.SysFont('consolas', 13).render(result.verdict, True, verdict_col)
            screen.blit(vs, (10, bottom_rect.top + 8))

            pen_txt = f'Penetrated {result.layers_penetrated}/{result.total_layers} layers'
            ps = pygame.font.SysFont('consolas', 12).render(pen_txt, True, (130, 150, 170))
            screen.blit(ps, (10, bottom_rect.top + 26))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
