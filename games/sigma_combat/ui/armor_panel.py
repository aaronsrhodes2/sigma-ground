"""
Armor panel — preset-first design.

Primary UI: scrollable card list of known armor systems grouped by era/type.
Custom builder: collapsible section at bottom with layer stack editor.
"""
import pygame
from .widgets import Button, font
from ..materials import MATERIAL_COLORS, MELTING_POINTS_K, MATERIAL_GROUPS
from ..presets import ARMOR_PRESETS, get_armor as _get_preset

MAX_LAYERS = 10

ARMOR_GROUPS = [
    ('HISTORICAL',       ['default', 'roman_lorica', 'medieval_plate', 'wwii_helmet']),
    ('MODERN BALLISTIC', ['nij_iiia', 'nij_iv', 'tank_composite', 'reactive_armor', 'bunker']),
    ('MARITIME',         ['wwii_battleship']),
    ('AEROSPACE',        ['aircraft_fuselage', 'spacecraft_hull']),
    ('SCI-FI / EXOTIC',  ['sci_fi_light', 'sci_fi_heavy', 'plasma_shielded', 'exotic_sigma']),
]

CARD_H  = 30
GROUP_H = 20

# Flattened material list for custom picker
_ALL_MATS = []
for _group, _mats in MATERIAL_GROUPS.items():
    for _m in _mats:
        _ALL_MATS.append((_m, _m.replace('_', ' ').title()))


class ArmorPanel:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self._selected_preset = 'default'
        self._custom_open     = False
        self._scroll_y        = 0
        self._max_scroll      = 0
        # Custom builder state
        self.layers           = list(ARMOR_PRESETS['default']['layers'])
        self._picking         = False
        self._picking_for     = -1
        self._build_widgets()

    def _build_widgets(self):
        x = self.rect.left + 6
        w = self.rect.width - 12
        self.add_btn     = Button((x, self.rect.bottom - 58, w // 2 - 3, 24),
                                  '+ Layer', color=(40, 80, 50))
        self.add_gap_btn = Button((x + w // 2 + 3, self.rect.bottom - 58,
                                   w // 2 - 3, 24), '+ Gap', color=(30, 60, 80))
        self.clear_btn   = Button((x, self.rect.bottom - 30, w, 24),
                                  'Clear All', color=(80, 40, 40))

    def get_layers(self):
        if not self._custom_open:
            return _get_preset(self._selected_preset)
        return list(self.layers)

    # ── List geometry ─────────────────────────────────────────────────────────

    def _list_area(self):
        return pygame.Rect(self.rect.left, self.rect.top,
                           self.rect.width, self.rect.height - 32)

    def _total_list_height(self):
        h = 0
        for _, keys in ARMOR_GROUPS:
            h += GROUP_H
            h += sum(1 for k in keys if k in ARMOR_PRESETS) * CARD_H
        return h

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, event):
        # Material picker takes priority
        if self._picking:
            return self._handle_picker(event)

        la = self._list_area()

        # Toggle button
        toggle_r = pygame.Rect(self.rect.left + 4, self.rect.bottom - 28,
                               self.rect.width - 8, 24)
        if event.type == pygame.MOUSEBUTTONDOWN and toggle_r.collidepoint(event.pos):
            self._custom_open = not self._custom_open
            return True

        if not self._custom_open:
            # Scroll
            if event.type == pygame.MOUSEWHEEL and la.collidepoint(pygame.mouse.get_pos()):
                self._scroll_y = max(0, min(self._max_scroll,
                                            self._scroll_y - event.y * 18))
                return True
            # Card click
            if event.type == pygame.MOUSEBUTTONDOWN and la.collidepoint(event.pos):
                y = la.top + 2 - self._scroll_y
                for group, keys in ARMOR_GROUPS:
                    y += GROUP_H
                    for key in keys:
                        if key not in ARMOR_PRESETS:
                            continue
                        card_r = pygame.Rect(la.left + 3, y, la.width - 6, CARD_H - 2)
                        if card_r.collidepoint(event.pos):
                            self._selected_preset = key
                            return True
                        y += CARD_H
        else:
            # Custom builder events
            if self.add_btn.handle_event(event):
                if len(self.layers) < MAX_LAYERS:
                    self.layers.append({'material': 'steel_mild', 'thickness_m': 0.010})
                return True
            if self.add_gap_btn.handle_event(event):
                if len(self.layers) < MAX_LAYERS:
                    self.layers.append({'gap': True, 'thickness_m': 0.050})
                return True
            if self.clear_btn.handle_event(event):
                self.layers.clear()
                return True

            for i, layer in enumerate(self.layers):
                lr = self._layer_rect(i)
                if not lr.collidepoint(getattr(event, 'pos', (-1, -1))):
                    continue
                rm_r = pygame.Rect(lr.right - 22, lr.top + 4, 18, 18)
                if event.type == pygame.MOUSEBUTTONDOWN and rm_r.collidepoint(event.pos):
                    self.layers.pop(i)
                    return True
                if not layer.get('gap'):
                    mat_r = pygame.Rect(lr.left + 4, lr.top + 4, lr.width - 50, 18)
                    if event.type == pygame.MOUSEBUTTONDOWN and mat_r.collidepoint(event.pos):
                        self._picking = True
                        self._picking_for = i
                        return True
                if event.type == pygame.MOUSEWHEEL and lr.collidepoint(pygame.mouse.get_pos()):
                    layer['thickness_m'] = max(0.001, min(2.0,
                                                layer['thickness_m'] + 0.001 * event.y))
                    return True

        return False

    def _handle_picker(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._picking = False
            return True
        if event.type == pygame.MOUSEBUTTONDOWN:
            pr = self._picker_rect()
            if not pr.collidepoint(event.pos):
                self._picking = False
                return True
            start_y = pr.top + 4
            for i, (key, _) in enumerate(_ALL_MATS):
                ir = pygame.Rect(pr.left + 2, start_y + i * 20, pr.width - 4, 20)
                if ir.collidepoint(event.pos):
                    if 0 <= self._picking_for < len(self.layers):
                        self.layers[self._picking_for]['material'] = key
                    self._picking = False
                    return True
        return False

    def _picker_rect(self):
        pw = self.rect.width - 10
        ph = min(len(_ALL_MATS) * 20 + 8, self.rect.height - 60)
        return pygame.Rect(self.rect.left + 5, self.rect.top + 40, pw, ph)

    def _layer_rect(self, i):
        lh = 32
        return pygame.Rect(self.rect.left + 4,
                           self.rect.top + 22 + i * (lh + 4),
                           self.rect.width - 8, lh)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surface):
        pygame.draw.rect(surface, (18, 22, 34), self.rect)

        la = self._list_area()

        if not self._custom_open:
            self._draw_preset_list(surface, la)
        else:
            self._draw_custom_builder(surface)

        # Toggle button
        toggle_r = pygame.Rect(self.rect.left + 4, self.rect.bottom - 28,
                               self.rect.width - 8, 24)
        label = 'CUSTOM BUILDER  ▲' if self._custom_open else '▼  CUSTOM BUILDER'
        tc = (80, 130, 220) if self._custom_open else (80, 100, 140)
        bg = (16, 20, 40) if self._custom_open else (18, 22, 34)
        pygame.draw.rect(surface, bg, toggle_r, border_radius=4)
        pygame.draw.rect(surface, tc, toggle_r, 1, border_radius=4)
        tl = font(11).render(label, True, tc)
        surface.blit(tl, (toggle_r.centerx - tl.get_width() // 2,
                           toggle_r.centery - tl.get_height() // 2))

    def _draw_preset_list(self, surface, la):
        clip = pygame.Surface((la.width, la.height))
        clip.fill((18, 22, 34))

        y       = 2 - self._scroll_y
        total_h = 0

        for group, keys in ARMOR_GROUPS:
            gh = font(9).render(group, True, (70, 90, 130))
            clip.blit(gh, (5, y + 3))
            pygame.draw.line(clip, (40, 55, 80),
                             (5, y + GROUP_H - 3), (la.width - 5, y + GROUP_H - 3), 1)
            y       += GROUP_H
            total_h += GROUP_H

            for key in keys:
                if key not in ARMOR_PRESETS:
                    continue
                preset   = ARMOR_PRESETS[key]
                selected = (key == self._selected_preset)
                layers_l = preset['layers']
                n_mats   = sum(1 for l in layers_l if not l.get('gap'))
                t_mm     = sum(l['thickness_m'] for l in layers_l) * 1000

                card_r = pygame.Rect(3, y, la.width - 6, CARD_H - 2)

                if selected:
                    pygame.draw.rect(clip, (18, 30, 60), card_r, border_radius=3)
                    pygame.draw.rect(clip, (60, 120, 230), card_r, 1, border_radius=3)
                else:
                    pygame.draw.rect(clip, (24, 30, 46), card_r, border_radius=3)

                # Layer color dots (up to 6)
                dot_x = 6
                for l in layers_l[:6]:
                    if l.get('gap'):
                        col = (40, 50, 65)
                    else:
                        col = MATERIAL_COLORS.get(l['material'], (120, 120, 120))
                    pygame.draw.rect(clip, col,
                                     pygame.Rect(dot_x, y + 2, 5, CARD_H - 6),
                                     border_radius=1)
                    dot_x += 7

                # Name
                name_col = (215, 230, 255) if selected else (185, 200, 225)
                name_s = font(11).render(preset['name'][:30], True, name_col)
                clip.blit(name_s, (dot_x + 3, y + 3))

                # Stats (layers / total thickness)
                stat = font(9).render(f'{n_mats}L  {t_mm:.0f}mm', True,
                                      (100, 140, 200) if selected else (70, 90, 120))
                clip.blit(stat, (la.width - stat.get_width() - 5,
                                 y + CARD_H - 2 - stat.get_height() - 2))

                y       += CARD_H
                total_h += CARD_H

        self._max_scroll = max(0, total_h - la.height + 20)
        surface.blit(clip, la.topleft)

        if self._max_scroll > 0:
            frac = self._scroll_y / self._max_scroll
            sb_h = max(24, int(la.height * la.height / (total_h + 10)))
            sb_y = la.top + int(frac * (la.height - sb_h))
            pygame.draw.rect(surface, (60, 80, 120),
                             pygame.Rect(la.right - 4, sb_y, 3, sb_h), border_radius=2)

        # Selected name label
        if self._selected_preset in ARMOR_PRESETS:
            sn = font(10).render(
                f'Selected: {ARMOR_PRESETS[self._selected_preset]["name"][:28]}',
                True, (100, 130, 200))
            surface.blit(sn, (self.rect.left + 6, self.rect.bottom - 44))

    def _draw_custom_builder(self, surface):
        hint = font(10).render('scroll = thickness  |  click material to change',
                               True, (60, 80, 100))
        surface.blit(hint, (self.rect.left + 6, self.rect.top + 6))

        for i, layer in enumerate(self.layers):
            lr     = self._layer_rect(i)
            is_gap = layer.get('gap', False)
            mat    = layer.get('material', 'iron') if not is_gap else 'gap'
            t_mm   = layer['thickness_m'] * 1000

            bg = (28, 38, 55) if not is_gap else (22, 30, 45)
            pygame.draw.rect(surface, bg, lr, border_radius=3)
            col = MATERIAL_COLORS.get(mat, (120, 120, 120))
            pygame.draw.rect(surface, col, pygame.Rect(lr.left, lr.top, 5, lr.height),
                             border_radius=2)
            pygame.draw.rect(surface, (50, 65, 90), lr, 1, border_radius=3)

            lbl = (font(12).render(f'gap  {t_mm:.0f} mm', True, (80, 100, 130))
                   if is_gap else
                   font(12).render(f'{mat.replace("_"," ")[:14]}  {t_mm:.1f} mm',
                                   True, (200, 215, 235)))
            surface.blit(lbl, (lr.left + 10, lr.centery - lbl.get_height() // 2))

            if not is_gap:
                mp = MELTING_POINTS_K.get(mat, 0)
                mp_s = font(10).render(f'{mp}K', True, (80, 100, 120))
                surface.blit(mp_s, (lr.right - 44, lr.centery - mp_s.get_height() // 2))

            rm_r = pygame.Rect(lr.right - 22, lr.top + 7, 16, 16)
            pygame.draw.rect(surface, (80, 40, 40), rm_r, border_radius=2)
            x_s = font(11).render('x', True, (200, 150, 150))
            surface.blit(x_s, (rm_r.centerx - x_s.get_width() // 2,
                                rm_r.centery - x_s.get_height() // 2))

        self.add_btn.draw(surface)
        self.add_gap_btn.draw(surface)
        self.clear_btn.draw(surface)

        if self._picking:
            self._draw_picker(surface)

    def _draw_picker(self, surface):
        pr = self._picker_rect()
        ov = pygame.Surface((pr.width, pr.height), pygame.SRCALPHA)
        ov.fill((15, 20, 35, 240))
        surface.blit(ov, pr.topleft)
        pygame.draw.rect(surface, (80, 100, 140), pr, 1)
        sy = pr.top + 4
        for i, (key, label) in enumerate(_ALL_MATS):
            ir = pygame.Rect(pr.left + 2, sy + i * 20, pr.width - 4, 20)
            if ir.bottom > pr.bottom:
                break
            col = MATERIAL_COLORS.get(key, (100, 100, 100))
            pygame.draw.rect(surface, col,
                             pygame.Rect(ir.left, ir.top, 4, ir.height))
            t = font(12).render(label, True, (200, 215, 235))
            surface.blit(t, (ir.left + 8, ir.centery - t.get_height() // 2))
            mp = font(10).render(f'{MELTING_POINTS_K.get(key, 0)}K', True, (80, 100, 120))
            surface.blit(mp, (ir.right - mp.get_width() - 4,
                              ir.centery - mp.get_height() // 2))
