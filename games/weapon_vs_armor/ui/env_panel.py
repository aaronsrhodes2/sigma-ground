"""
Environment bar — top strip: environment selector + distance.
Default: Earth atmosphere at sea level (3m ASL).
"""
import pygame
from .widgets import Button, Slider, font
from ..physics.environment import ENVIRONMENTS

ENV_KEYS = list(ENVIRONMENTS.keys())
ENV_LABELS = [(k, ENVIRONMENTS[k]['label']) for k in ENV_KEYS]


class EnvPanel:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        # Default: Earth atmosphere (index 1)
        self.selected_env = 1  # atmosphere_earth
        self.distance_m = 100.0

        bw = 140
        bh = 28
        bx = self.rect.left + 6
        by = self.rect.top + (self.rect.height - bh) // 2

        self.env_btns = []
        for i, (key, label) in enumerate(ENV_LABELS):
            btn = Button((bx + i * (bw + 4), by, bw, bh),
                         label[:20],
                         color=(30, 45, 65),
                         active=(i == self.selected_env),
                         active_color=(50, 90, 130))
            self.env_btns.append(btn)

        # Distance slider
        slider_x = bx + len(ENV_LABELS) * (bw + 4) + 20
        self.dist_slider = Slider(
            (slider_x, by + 4, 200, 18),
            min_val=1, max_val=10000, initial=100.0, log=True,
            label='Range',
            format_fn=lambda v: f'{v:.0f} m',
            ticks=[(10,'10m'), (100,'100m'), (1000,'1km'), (5000,'5km')]
        )

    @property
    def environment(self):
        return ENV_KEYS[self.selected_env]

    def handle_event(self, event):
        for i, btn in enumerate(self.env_btns):
            if btn.handle_event(event):
                self.selected_env = i
                for j, b in enumerate(self.env_btns):
                    b.active = (j == i)
                return True

        if self.dist_slider.handle_event(event):
            self.distance_m = self.dist_slider.value
            return True

        return False

    def draw(self, surface):
        pygame.draw.rect(surface, (15, 20, 30), self.rect)
        pygame.draw.rect(surface, (45, 60, 85), self.rect, 1)

        for btn in self.env_btns:
            btn.draw(surface)

        self.dist_slider.draw(surface)

        # Description of selected env
        env_desc = ENVIRONMENTS[self.environment]['description']
        d = font(10).render(env_desc[:80], True, (70, 90, 115))
        surface.blit(d, (self.rect.left + 6, self.rect.bottom - 15))
