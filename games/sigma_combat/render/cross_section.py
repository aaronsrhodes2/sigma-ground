"""
2D cross-section renderer — draws the armor stack and weapon damage channel.

The view is a top-down horizontal cross-section:
  [WEAPON] ---> [Layer1][Layer2][Layer3]...

Layers are colored by material with damage heat-overlay.
Damage channel is drawn as a tapered cone/cylinder.
"""

import math
import pygame
from ..materials import temperature_color, MATERIAL_COLORS, MELTING_POINTS_K

# Fonts — initialized lazily
_fonts = {}


def _font(size=16):
    if size not in _fonts:
        _fonts[size] = pygame.font.SysFont('consolas', size)
    return _fonts[size]


# Layout constants
WEAPON_ZONE_W = 80   # px on left reserved for weapon icon + arrow
MIN_LAYER_W = 18     # minimum px width per layer (so thin layers are still visible)
LAYER_HEIGHT_FRAC = 0.65  # fraction of panel height for the armor block
GAP_COLOR = (30, 35, 45)
GAP_STRIPE = (40, 50, 65)


def render(surface: pygame.Surface,
           armor_layers: list,
           result=None,
           weapon_type: str = 'kinetic',
           title: str = '') -> None:
    """
    Draw the cross-section onto *surface* (clears it first).

    armor_layers: list of {'material':str, 'thickness_m':float}
                  or {'gap':True, 'thickness_m':float}
    result: DamageResult from solver.solve(), or None (design mode)
    """
    W, H = surface.get_size()
    surface.fill((18, 22, 30))

    if title:
        t = _font(14).render(title, True, (120, 140, 160))
        surface.blit(t, (8, 4))

    # Filter to renderable layers (skip empty armor)
    layers = [l for l in armor_layers if l]

    if not layers:
        msg = _font(18).render('No armor layers defined', True, (80, 90, 100))
        surface.blit(msg, (W // 2 - msg.get_width() // 2, H // 2))
        return

    # Compute pixel widths
    total_thickness = sum(l['thickness_m'] for l in layers)
    available_w = W - WEAPON_ZONE_W - 20
    total_thickness = max(total_thickness, 1e-10)

    # Scale: each layer width proportional to thickness, minimum MIN_LAYER_W
    raw_widths = [max(MIN_LAYER_W, int(l['thickness_m'] / total_thickness * available_w))
                  for l in layers]
    # Rescale to fit exactly
    scale = available_w / max(sum(raw_widths), 1)
    px_widths = [max(MIN_LAYER_W, int(w * scale)) for w in raw_widths]

    # Layer block bounds
    block_h = int(H * LAYER_HEIGHT_FRAC)
    block_y = (H - block_h) // 2 + 15
    x = WEAPON_ZONE_W

    layer_rects = []
    for i, (layer, pw) in enumerate(zip(layers, px_widths)):
        rect = pygame.Rect(x, block_y, pw, block_h)
        layer_rects.append(rect)
        x += pw

    # ── Draw weapon entry zone ────────────────────────────────────
    _draw_weapon_zone(surface, WEAPON_ZONE_W, block_y, block_h, weapon_type)

    # ── Draw layers ───────────────────────────────────────────────
    for i, (layer, rect) in enumerate(zip(layers, layer_rects)):
        if layer.get('gap', False):
            _draw_gap(surface, rect)
            continue

        mat = layer['material']
        base_color = MATERIAL_COLORS.get(mat, (120, 120, 120))

        # Determine temperature for this layer (from result if available)
        temp_k = 300.0
        layer_state = None
        if result and i < len(result.layers):
            layer_state = result.layers[i]
            temp_k = layer_state.temperature_k

        draw_color = temperature_color(mat, temp_k) if result else base_color

        # Draw layer body
        pygame.draw.rect(surface, draw_color, rect)
        pygame.draw.rect(surface, (0, 0, 0), rect, 1)  # border

        # Damage channel overlay
        if result and layer_state:
            _draw_damage_channel(surface, rect, layer_state, draw_color)

        # Label
        _draw_layer_label(surface, rect, layer, temp_k, layer_state)

    # ── Draw penetration indicator ────────────────────────────────
    if result:
        _draw_penetration_summary(surface, layer_rects, result, block_y, block_h)

    # ── Thickness ruler ───────────────────────────────────────────
    _draw_ruler(surface, layers, layer_rects, block_y + block_h + 5)


def _draw_weapon_zone(surface, zone_w, block_y, block_h, weapon_type):
    WCOLORS = {
        'kinetic': (200, 180, 80),
        'laser': (80, 200, 255),
        'plasma': (255, 120, 40),
        'explosive': (255, 60, 60),
        'particle': (140, 255, 140),
        'gravitational': (180, 100, 255),
    }
    color = WCOLORS.get(weapon_type, (200, 200, 200))
    cx = zone_w // 2
    mid_y = block_y + block_h // 2

    # Arrow
    pygame.draw.line(surface, color, (10, mid_y), (zone_w - 8, mid_y), 3)
    # Arrowhead
    pts = [(zone_w - 8, mid_y),
           (zone_w - 18, mid_y - 8),
           (zone_w - 18, mid_y + 8)]
    pygame.draw.polygon(surface, color, pts)

    # Type label
    label = _font(11).render(weapon_type.upper()[:7], True, color)
    surface.blit(label, (cx - label.get_width() // 2, mid_y - 25))


def _draw_gap(surface, rect):
    pygame.draw.rect(surface, GAP_COLOR, rect)
    # Dashed stripe pattern
    for y in range(rect.top, rect.bottom, 8):
        pygame.draw.line(surface, GAP_STRIPE, (rect.left, y), (rect.right, y), 1)
    pygame.draw.rect(surface, (50, 60, 80), rect, 1)
    label = _font(10).render('GAP', True, (80, 100, 120))
    if rect.width >= label.get_width():
        surface.blit(label, (rect.centerx - label.get_width() // 2,
                              rect.centery - label.get_height() // 2))


def _draw_damage_channel(surface, rect, layer_state, base_color):
    if layer_state.damage_fraction <= 0:
        return

    depth_px = int(rect.width * layer_state.damage_fraction)
    channel_h = int(rect.height * 0.4)
    cy = rect.centery

    # Heat gradient overlay
    if depth_px > 0:
        dmg_rect = pygame.Rect(rect.left, cy - channel_h // 2, depth_px, channel_h)
        # Color based on temperature
        T = layer_state.temperature_k
        if T > 3000:
            overlay = (255, 255, 200, 180)
        elif T > 1500:
            overlay = (255, 140, 30, 160)
        elif T > 800:
            overlay = (200, 80, 20, 120)
        else:
            overlay = (150, 50, 50, 80)

        heat_surf = pygame.Surface((depth_px, channel_h), pygame.SRCALPHA)
        heat_surf.fill(overlay)
        surface.blit(heat_surf, dmg_rect.topleft)

        # Penetration arrow if fully through
        if layer_state.penetrated:
            pygame.draw.line(surface,
                             (255, 255, 100),
                             (rect.left + 2, cy),
                             (rect.right - 2, cy), 2)


def _draw_layer_label(surface, rect, layer, temp_k, layer_state):
    mat = layer.get('material', '?')
    thick_mm = layer['thickness_m'] * 1000
    lines = [mat[:10], f'{thick_mm:.1f}mm']

    if layer_state and temp_k > 350:
        lines.append(f'{temp_k:.0f}K')

    font = _font(11)
    total_h = len(lines) * 13
    start_y = rect.centery - total_h // 2

    for j, line in enumerate(lines):
        surf = font.render(line, True, (230, 230, 230))
        if surf.get_width() < rect.width - 4:
            surface.blit(surf, (rect.centerx - surf.get_width() // 2,
                                start_y + j * 13))
        else:
            # Too narrow — just first char
            surf = font.render(mat[0].upper(), True, (200, 200, 200))
            surface.blit(surf, (rect.centerx - surf.get_width() // 2, rect.centery - 6))
            break


def _draw_penetration_summary(surface, layer_rects, result, block_y, block_h):
    if not result or not layer_rects:
        return

    # Vertical marker at stopping point
    stopped_layer = result.layers_penetrated
    if stopped_layer < len(layer_rects):
        rect = layer_rects[stopped_layer]
        state = result.layers[stopped_layer] if stopped_layer < len(result.layers) else None
        depth_frac = state.damage_fraction if state else 0.0
        stop_x = rect.left + int(rect.width * depth_frac)
        color = (255, 80, 80) if not result.breached else (255, 220, 50)
        pygame.draw.line(surface, color,
                         (stop_x, block_y - 6),
                         (stop_x, block_y + block_h + 6), 2)

        label_text = 'STOPPED' if not result.breached else 'BREACHED'
        label = _font(12).render(label_text, True, color)
        lx = max(0, stop_x - label.get_width() // 2)
        surface.blit(label, (lx, block_y - 20))


def _draw_ruler(surface, layers, layer_rects, y):
    if not layer_rects:
        return
    total_mm = sum(l['thickness_m'] * 1000 for l in layers if not l.get('gap'))
    font = _font(10)
    col = (80, 100, 120)

    # Left edge
    pygame.draw.line(surface, col,
                     (layer_rects[0].left, y),
                     (layer_rects[-1].right, y), 1)
    # Ticks + total
    for rect, layer in zip(layer_rects, layers):
        if not layer.get('gap'):
            pygame.draw.line(surface, col, (rect.left, y), (rect.left, y + 4), 1)
    pygame.draw.line(surface, col,
                     (layer_rects[-1].right, y),
                     (layer_rects[-1].right, y + 4), 1)

    total_label = font.render(f'Total: {total_mm:.1f} mm', True, col)
    cx = (layer_rects[0].left + layer_rects[-1].right) // 2
    surface.blit(total_label, (cx - total_label.get_width() // 2, y + 6))
