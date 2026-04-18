"""
Reusable pygame UI widgets: Slider, Button, Dropdown, Label.
"""
import math
import pygame

_fonts = {}


def font(size=14):
    if size not in _fonts:
        _fonts[size] = pygame.font.SysFont('consolas', size)
    return _fonts[size]


def lerp(a, b, t):
    return a + (b - a) * t


class Slider:
    """Horizontal slider with optional log scale and reference tick labels."""

    def __init__(self, rect, min_val, max_val, initial,
                 label='', log=False, format_fn=None, ticks=None):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.log = log
        self.format_fn = format_fn or (lambda v: f'{v:.3g}')
        self.ticks = ticks or []   # list of (value, label_str)
        self.dragging = False
        self.track_h = 6
        self.thumb_r = 9

    @property
    def _track_rect(self):
        cy = self.rect.centery
        return pygame.Rect(self.rect.left + self.thumb_r,
                           cy - self.track_h // 2,
                           self.rect.width - 2 * self.thumb_r,
                           self.track_h)

    def _val_to_x(self, v):
        tr = self._track_rect
        if self.log:
            if self.min_val <= 0:
                t = 0.0
            else:
                log_min = math.log10(self.min_val)
                log_max = math.log10(self.max_val)
                t = (math.log10(max(v, self.min_val)) - log_min) / (log_max - log_min)
        else:
            t = (v - self.min_val) / (self.max_val - self.min_val)
        return tr.left + int(t * tr.width)

    def _x_to_val(self, x):
        tr = self._track_rect
        t = max(0.0, min(1.0, (x - tr.left) / tr.width))
        if self.log:
            log_min = math.log10(max(self.min_val, 1e-30))
            log_max = math.log10(self.max_val)
            return 10 ** (log_min + t * (log_max - log_min))
        else:
            return self.min_val + t * (self.max_val - self.min_val)

    def draw(self, surface, label_color=(180, 190, 200), highlight=False):
        tr = self._track_rect
        cy = self.rect.centery

        # Track background
        pygame.draw.rect(surface, (40, 50, 65), tr, border_radius=3)

        # Filled portion
        fill_x = self._val_to_x(self.value)
        fill_rect = pygame.Rect(tr.left, tr.top, fill_x - tr.left, tr.height)
        if fill_rect.width > 0:
            pygame.draw.rect(surface, (60, 130, 200), fill_rect, border_radius=3)

        # Thumb
        thumb_color = (200, 220, 255) if highlight or self.dragging else (140, 170, 210)
        pygame.draw.circle(surface, thumb_color, (fill_x, cy), self.thumb_r)
        pygame.draw.circle(surface, (80, 100, 140), (fill_x, cy), self.thumb_r, 1)

        # Tick labels
        for tick_v, tick_lbl in self.ticks:
            tx = self._val_to_x(tick_v)
            pygame.draw.line(surface, (70, 90, 110),
                             (tx, cy + self.thumb_r),
                             (tx, cy + self.thumb_r + 4), 1)
            t_surf = font(9).render(tick_lbl, True, (80, 100, 120))
            surface.blit(t_surf, (tx - t_surf.get_width() // 2,
                                   cy + self.thumb_r + 6))

        # Label + value
        if self.label:
            lbl = font(12).render(self.label, True, label_color)
            surface.blit(lbl, (self.rect.left, self.rect.top - 16))

        val_str = self.format_fn(self.value)
        val_surf = font(12).render(val_str, True, (220, 230, 245))
        surface.blit(val_surf, (self.rect.right - val_surf.get_width(),
                                 self.rect.top - 16))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.value = self._x_to_val(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.value = self._x_to_val(event.pos[0])
                return True
        return False


class Button:
    def __init__(self, rect, text, color=(60, 100, 160), text_color=(220, 230, 245),
                 active=False, active_color=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.active = active
        self.active_color = active_color or (80, 150, 220)
        self.hovered = False

    def draw(self, surface):
        col = self.active_color if self.active else (
            tuple(min(255, c + 25) for c in self.color) if self.hovered else self.color
        )
        pygame.draw.rect(surface, col, self.rect, border_radius=4)
        pygame.draw.rect(surface, (80, 100, 140), self.rect, 1, border_radius=4)
        txt = font(13).render(self.text, True, self.text_color)
        surface.blit(txt, (self.rect.centerx - txt.get_width() // 2,
                            self.rect.centery - txt.get_height() // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Dropdown:
    def __init__(self, rect, options, selected=0, label=''):
        self.rect = pygame.Rect(rect)
        self.options = options   # list of (key, display_label)
        self.selected = selected
        self.label = label
        self.open = False
        self._item_h = 22

    @property
    def value(self):
        return self.options[self.selected][0]

    def draw(self, surface):
        # Label
        if self.label:
            lbl = font(12).render(self.label, True, (160, 175, 190))
            surface.blit(lbl, (self.rect.left, self.rect.top - 16))

        # Box
        col = (50, 65, 90) if not self.open else (70, 90, 120)
        pygame.draw.rect(surface, col, self.rect, border_radius=3)
        pygame.draw.rect(surface, (80, 100, 140), self.rect, 1, border_radius=3)
        txt = font(12).render(self.options[self.selected][1], True, (210, 225, 240))
        surface.blit(txt, (self.rect.left + 6, self.rect.centery - txt.get_height() // 2))
        # Arrow
        ax = self.rect.right - 14
        ay = self.rect.centery
        pts = [(ax - 5, ay - 3), (ax + 5, ay - 3), (ax, ay + 4)]
        pygame.draw.polygon(surface, (150, 170, 200), pts)

        # Dropdown list
        if self.open:
            list_rect = pygame.Rect(self.rect.left,
                                    self.rect.bottom,
                                    self.rect.width,
                                    self._item_h * len(self.options))
            pygame.draw.rect(surface, (35, 45, 65), list_rect)
            pygame.draw.rect(surface, (80, 100, 140), list_rect, 1)
            for i, (_, lbl_text) in enumerate(self.options):
                item_rect = pygame.Rect(self.rect.left,
                                        self.rect.bottom + i * self._item_h,
                                        self.rect.width, self._item_h)
                if i == self.selected:
                    pygame.draw.rect(surface, (60, 90, 140), item_rect)
                item_txt = font(12).render(lbl_text, True, (200, 215, 235))
                surface.blit(item_txt, (item_rect.left + 6,
                                         item_rect.centery - item_txt.get_height() // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
                return False
            if self.open:
                for i in range(len(self.options)):
                    item_rect = pygame.Rect(self.rect.left,
                                            self.rect.bottom + i * self._item_h,
                                            self.rect.width, self._item_h)
                    if item_rect.collidepoint(event.pos):
                        self.selected = i
                        self.open = False
                        return True
                self.open = False
        return False
