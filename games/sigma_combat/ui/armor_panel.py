"""
Armor builder panel — right side of screen.
Lets user add/remove layers and set thickness.
"""
import pygame
from .widgets import Button, Slider, Dropdown, font
from ..materials import all_materials, MATERIAL_DESCRIPTIONS, MATERIAL_COLORS, MELTING_POINTS_K, MATERIAL_GROUPS

MAX_LAYERS = 10

# Flattened ordered material list for the picker
_ALL_MATS = []
for group, mats in MATERIAL_GROUPS.items():
    for m in mats:
        _ALL_MATS.append((m, f'{m.replace("_", " ").title()}'))


class ArmorPanel:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        # Default armor: titanium + tungsten
        self.layers = [
            {'material': 'titanium', 'thickness_m': 0.015},
            {'material': 'tungsten', 'thickness_m': 0.010},
        ]
        self._build_widgets()
        self._picking = False  # material picker open?
        self._picking_for = -1  # which layer index

    def _build_widgets(self):
        x = self.rect.left + 6
        w = self.rect.width - 12
        self.add_btn = Button((x, self.rect.bottom - 58, w // 2 - 3, 24),
                              '+ Layer', color=(40, 80, 50))
        self.add_gap_btn = Button((x + w // 2 + 3, self.rect.bottom - 58, w // 2 - 3, 24),
                                  '+ Gap', color=(30, 60, 80))
        self.clear_btn = Button((x, self.rect.bottom - 30, w, 24),
                                'Clear All', color=(80, 40, 40))

    def get_layers(self):
        return list(self.layers)

    def handle_event(self, event):
        if self._picking:
            return self._handle_picker(event)

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

        # Per-layer controls
        for i, layer in enumerate(self.layers):
            lr = self._layer_rect(i)
            if not lr.collidepoint(getattr(event, 'pos', (-1, -1))):
                continue

            # Remove button
            rm_r = pygame.Rect(lr.right - 22, lr.top + 4, 18, 18)
            if event.type == pygame.MOUSEBUTTONDOWN and rm_r.collidepoint(event.pos):
                self.layers.pop(i)
                return True

            # Material button (for non-gaps)
            if not layer.get('gap'):
                mat_r = pygame.Rect(lr.left + 4, lr.top + 4, lr.width - 50, 18)
                if event.type == pygame.MOUSEBUTTONDOWN and mat_r.collidepoint(event.pos):
                    self._picking = True
                    self._picking_for = i
                    return True

            # Thickness change (wheel scroll)
            if event.type == pygame.MOUSEWHEEL and lr.collidepoint(pygame.mouse.get_pos()):
                delta = 0.001 * event.y
                layer['thickness_m'] = max(0.001, min(2.0, layer['thickness_m'] + delta))
                return True

            # Drag thickness (horizontal drag in the layer row)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if lr.collidepoint(event.pos):
                    layer['thickness_m'] = max(0.001, layer['thickness_m'] * 0.5)
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
            # Check item click
            item_h = 20
            start_y = pr.top + 4
            for i, (key, label) in enumerate(_ALL_MATS):
                item_r = pygame.Rect(pr.left + 2, start_y + i * item_h, pr.width - 4, item_h)
                if item_r.collidepoint(event.pos):
                    if 0 <= self._picking_for < len(self.layers):
                        self.layers[self._picking_for]['material'] = key
                    self._picking = False
                    return True
        return False

    def _picker_rect(self):
        pw = self.rect.width - 10
        ph = min(len(_ALL_MATS) * 20 + 8, self.rect.height - 60)
        px = self.rect.left + 5
        py = self.rect.top + 40
        return pygame.Rect(px, py, pw, ph)

    def _layer_rect(self, i):
        lh = 32
        return pygame.Rect(self.rect.left + 4,
                           self.rect.top + 22 + i * (lh + 4),
                           self.rect.width - 8, lh)

    def draw(self, surface):
        pygame.draw.rect(surface, (20, 26, 38), self.rect)
        pygame.draw.rect(surface, (50, 65, 90), self.rect, 1)

        title = font(14).render('ARMOR LAYERS', True, (140, 160, 200))
        surface.blit(title, (self.rect.left + 6, self.rect.top - 20))

        hint = font(10).render('scroll = thickness  |  click material to change', True, (60, 80, 100))
        surface.blit(hint, (self.rect.left + 6, self.rect.top + 6))

        for i, layer in enumerate(self.layers):
            lr = self._layer_rect(i)
            is_gap = layer.get('gap', False)
            mat = layer.get('material', 'iron') if not is_gap else 'gap'
            thick_mm = layer['thickness_m'] * 1000

            # Row bg
            bg_color = (28, 38, 55) if not is_gap else (22, 30, 45)
            pygame.draw.rect(surface, bg_color, lr, border_radius=3)
            mat_color = MATERIAL_COLORS.get(mat, (120, 120, 120))
            pygame.draw.rect(surface, mat_color,
                             pygame.Rect(lr.left, lr.top, 5, lr.height),
                             border_radius=2)
            pygame.draw.rect(surface, (50, 65, 90), lr, 1, border_radius=3)

            # Layer label
            if is_gap:
                lbl = font(12).render(f'gap  {thick_mm:.0f} mm', True, (80, 100, 130))
            else:
                short = mat.replace('_', ' ')[:14]
                lbl = font(12).render(f'{short}  {thick_mm:.1f} mm', True, (200, 215, 235))
            surface.blit(lbl, (lr.left + 10, lr.centery - lbl.get_height() // 2))

            # Melting point hint
            if not is_gap:
                mp = MELTING_POINTS_K.get(mat, 0)
                mp_lbl = font(10).render(f'{mp}K', True, (80, 100, 120))
                surface.blit(mp_lbl, (lr.right - 44, lr.centery - mp_lbl.get_height() // 2))

            # Remove button
            rm_r = pygame.Rect(lr.right - 22, lr.top + 7, 16, 16)
            pygame.draw.rect(surface, (80, 40, 40), rm_r, border_radius=2)
            x_lbl = font(11).render('x', True, (200, 150, 150))
            surface.blit(x_lbl, (rm_r.centerx - x_lbl.get_width() // 2,
                                   rm_r.centery - x_lbl.get_height() // 2))

        self.add_btn.draw(surface)
        self.add_gap_btn.draw(surface)
        self.clear_btn.draw(surface)

        # Material picker overlay
        if self._picking:
            self._draw_picker(surface)

    def _draw_picker(self, surface):
        pr = self._picker_rect()
        overlay = pygame.Surface((pr.width, pr.height), pygame.SRCALPHA)
        overlay.fill((15, 20, 35, 240))
        surface.blit(overlay, pr.topleft)
        pygame.draw.rect(surface, (80, 100, 140), pr, 1)

        item_h = 20
        start_y = pr.top + 4
        for i, (key, label) in enumerate(_ALL_MATS):
            item_r = pygame.Rect(pr.left + 2, start_y + i * item_h, pr.width - 4, item_h)
            if item_r.bottom > pr.bottom:
                break
            col = MATERIAL_COLORS.get(key, (100, 100, 100))
            pygame.draw.rect(surface, col,
                             pygame.Rect(item_r.left, item_r.top, 4, item_r.height))
            txt = font(12).render(label, True, (200, 215, 235))
            surface.blit(txt, (item_r.left + 8, item_r.centery - txt.get_height() // 2))

            # Melting point
            mp_k = MELTING_POINTS_K.get(key, 0)
            mp_t = font(10).render(f'{mp_k}K', True, (80, 100, 120))
            surface.blit(mp_t, (item_r.right - mp_t.get_width() - 4,
                                 item_r.centery - mp_t.get_height() // 2))
