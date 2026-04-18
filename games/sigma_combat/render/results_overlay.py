"""
Results overlay — draws the post-fire verdict panel onto the screen.
"""
import pygame

_fonts = {}


def _font(size):
    if size not in _fonts:
        _fonts[size] = pygame.font.SysFont('consolas', size)
    return _fonts[size]


def render(surface: pygame.Surface, result, rect: pygame.Rect) -> None:
    """Draw results inside rect on the given surface."""
    if result is None:
        return

    # Semi-transparent background
    bg = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg.fill((10, 12, 18, 220))
    surface.blit(bg, rect.topleft)
    pygame.draw.rect(surface, (60, 70, 90), rect, 1)

    y = rect.top + 8
    x = rect.left + 10

    # Verdict headline
    breach_color = (255, 80, 80) if result.breached else (80, 255, 120)
    verdict_surf = _font(18).render(result.verdict, True, breach_color)
    surface.blit(verdict_surf, (x, y))
    y += 26

    # Divider
    pygame.draw.line(surface, (60, 70, 90), (x, y), (rect.right - 10, y), 1)
    y += 6

    # Per-layer summary
    for i, (line, state) in enumerate(zip(result.summary_lines, result.layers)):
        if state.material == 'gap':
            continue
        # Truncate long lines
        if len(line) > 90:
            line = line[:87] + '...'
        col = (255, 120, 120) if state.penetrated else (160, 200, 160)
        surf = _font(12).render(line, True, col)
        surface.blit(surf, (x + 4, y))
        y += 15
        if y > rect.bottom - 20:
            more = _font(11).render('...more layers...', True, (80, 90, 100))
            surface.blit(more, (x + 4, y))
            break

    # Footer
    y = rect.bottom - 20
    env_txt = f'Environment: {result.environment}  |  Weapon: {result.weapon_type}'
    footer = _font(11).render(env_txt, True, (70, 85, 100))
    surface.blit(footer, (x, y))
