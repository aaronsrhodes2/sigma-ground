"""
Slow-motion impact animation — ballistics-gel / high-speed-camera style.

Phases (total ~240 frames at 60fps = 4 seconds):
  0–40    APPROACH   weapon travels from left to first layer surface
  40–180  PENETRATE  weapon moves through each layer in sequence;
                     per-layer: shockwave bloom, heat halo, cracks/spall particles
  180–210 SETTLE     weapon embedded (stopped) or exits (breached) + debris coast
  210–240 HOLD       static result freeze-frame before handing off to normal render
"""
import math
import random
import pygame

APPROACH_START  = 0
APPROACH_END    = 40
PENETRATE_END   = 180
SETTLE_END      = 210
TOTAL_FRAMES    = 240

WEAPON_COLORS = {
    'kinetic':      (220, 200,  80),
    'laser':        ( 80, 220, 255),
    'plasma':       (255, 140,  50),
    'explosive':    (255,  80,  40),
    'particle':     (140, 255, 140),
    'gravitational':(180, 100, 255),
}

# Shockwave ring colors per weapon type
SHOCK_COLORS = {
    'kinetic':      (255, 220, 100),
    'laser':        (100, 240, 255),
    'plasma':       (255, 160,  60),
    'explosive':    (255, 120,  40),
    'particle':     (180, 255, 180),
    'gravitational':(200, 130, 255),
}


# ── Particle ─────────────────────────────────────────────────────────────────

class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size', 'gravity')

    def __init__(self, x, y, vx, vy, life, color, size=2, gravity=0.12):
        self.x = x;  self.y = y
        self.vx = vx; self.vy = vy
        self.life = life; self.max_life = life
        self.color = color; self.size = size
        self.gravity = gravity

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vx *= 0.94
        self.vy *= 0.94
        self.vy += self.gravity
        self.life -= 1

    @property
    def alpha(self):
        return self.life / max(self.max_life, 1)

    def draw(self, surface):
        a = self.alpha
        r = min(255, int(self.color[0] * a))
        g = min(255, int(self.color[1] * a))
        b = min(255, int(self.color[2] * a))
        sz = max(1, self.size)
        pygame.draw.circle(surface, (r, g, b), (int(self.x), int(self.y)), sz)


# ── Shockwave ring ────────────────────────────────────────────────────────────

class ShockRing:
    __slots__ = ('x', 'y', 'radius', 'max_radius', 'life', 'max_life', 'color', 'width')

    def __init__(self, x, y, max_radius, life, color, width=2):
        self.x = x; self.y = y
        self.radius = 0; self.max_radius = max_radius
        self.life = life; self.max_life = life
        self.color = color; self.width = width

    def update(self):
        self.life -= 1
        t = 1.0 - self.life / max(self.max_life, 1)
        self.radius = int(self.max_radius * (t ** 0.6))

    @property
    def alive(self):
        return self.life > 0

    def draw(self, surface):
        if self.radius < 1:
            return
        a = (self.life / max(self.max_life, 1)) ** 0.5
        r = min(255, int(self.color[0] * a))
        g = min(255, int(self.color[1] * a))
        b = min(255, int(self.color[2] * a))
        pygame.draw.circle(surface, (r, g, b), (int(self.x), int(self.y)),
                           int(self.radius), self.width)


# ── Heat bloom (expanding translucent ellipse) ────────────────────────────────

class HeatBloom:
    __slots__ = ('x', 'y', 'rw', 'rh', 'max_rw', 'max_rh', 'life', 'max_life', 'color')

    def __init__(self, x, y, max_rw, max_rh, life, color):
        self.x = x; self.y = y
        self.rw = 0; self.rh = 0
        self.max_rw = max_rw; self.max_rh = max_rh
        self.life = life; self.max_life = life
        self.color = color

    def update(self):
        self.life -= 1
        t = 1.0 - self.life / max(self.max_life, 1)
        ease = t ** 0.5
        self.rw = int(self.max_rw * ease)
        self.rh = int(self.max_rh * ease)

    @property
    def alive(self):
        return self.life > 0

    def draw(self, surface):
        if self.rw < 1 or self.rh < 1:
            return
        a = int(100 * (self.life / max(self.max_life, 1)) ** 1.5)
        if a < 2:
            return
        s = pygame.Surface((self.rw * 2 + 2, self.rh * 2 + 2), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*self.color, a),
                            (0, 0, self.rw * 2, self.rh * 2))
        surface.blit(s, (int(self.x) - self.rw, int(self.y) - self.rh))


# ── AnimationState ────────────────────────────────────────────────────────────

class AnimationState:
    def __init__(self, result, weapon: dict, layer_rects: list,
                 weapon_zone_x: int, panel_h: int):
        self.result       = result
        self.weapon_type  = weapon.get('type', 'kinetic')
        self.weapon       = weapon
        self.layer_rects  = list(layer_rects)
        self.weapon_zone_x = weapon_zone_x
        self.panel_h      = panel_h
        self.frame        = 0
        self.done         = False

        self.particles:  list[Particle]  = []
        self.rings:      list[ShockRing] = []
        self.blooms:     list[HeatBloom] = []

        self._breach_flash = 0

        # Only animate layers the weapon actually reaches:
        #   - all penetrated layers + the stopping layer (or all if breached)
        if result and not result.breached:
            n_active = min(result.layers_penetrated + 1, len(layer_rects))
        else:
            n_active = len(layer_rects)
        self._n_active = n_active
        self._layer_triggered = [False] * max(n_active, 1)

        # Allocate penetration time proportional to resistance of each active layer
        if n_active > 0:
            weights = []
            for i in range(n_active):
                st = result.layers[i] if (result and i < len(result.layers)) else None
                w = (st.damage_fraction if st else 0.5) + 0.15
                weights.append(w)
            total_w = sum(weights) or 1.0
            cum = 0.0
            self._layer_trigger_t = []
            for w in weights:
                self._layer_trigger_t.append(cum / total_w)
                cum += w
        else:
            self._layer_trigger_t = []

    # ── Phase helpers ─────────────────────────────────────────────────────────

    @property
    def approach_t(self):
        if self.frame >= APPROACH_END:
            return 1.0
        return self.frame / APPROACH_END

    @property
    def penetrate_t(self):
        if self.frame <= APPROACH_END:
            return 0.0
        if self.frame >= PENETRATE_END:
            return 1.0
        return (self.frame - APPROACH_END) / (PENETRATE_END - APPROACH_END)

    @property
    def settle_t(self):
        if self.frame <= PENETRATE_END:
            return 0.0
        if self.frame >= SETTLE_END:
            return 1.0
        return (self.frame - PENETRATE_END) / (SETTLE_END - PENETRATE_END)

    # ── Weapon position during penetration ───────────────────────────────────

    def _weapon_x(self):
        """Return current tip x-coordinate of the weapon/beam head."""
        if not self.layer_rects:
            return self.weapon_zone_x

        first = self.layer_rects[0]
        last  = self.layer_rects[-1]

        if self.frame < APPROACH_END:
            # Ease-in from weapon zone edge to first layer
            t = self.approach_t ** 1.5
            return int(self.weapon_zone_x + (first.left - self.weapon_zone_x - 6) * t)

        if self.frame >= SETTLE_END:
            return self._final_x()

        pt = self.penetrate_t

        # Walk only the active layers (weapon can't enter layers beyond stopping point)
        for i, trig in enumerate(self._layer_trigger_t):
            next_t = self._layer_trigger_t[i + 1] if i + 1 < len(self._layer_trigger_t) else 1.0
            if pt <= next_t or i == self._n_active - 1:
                rect = self.layer_rects[i]
                st   = self.result.layers[i] if (self.result and i < len(self.result.layers)) else None
                df   = st.damage_fraction if st else 1.0
                span = next_t - trig
                local_t = (pt - trig) / span if span > 0 else 1.0
                local_t = min(1.0, local_t)
                if st and not st.penetrated:
                    # Decelerate into stopping layer — ease-out so it visually brakes
                    x = rect.left + int(rect.width * df * (local_t ** 2))
                else:
                    x = rect.left + int(rect.width * local_t)
                return x

        return self._final_x()

    def _final_x(self):
        """Resting / exit x after animation."""
        if not self.result or not self.layer_rects:
            return self.weapon_zone_x
        n_pen = self.result.layers_penetrated
        if self.result.breached:
            return self.layer_rects[-1].right + 20
        if n_pen < len(self.layer_rects):
            rect = self.layer_rects[n_pen]
            st   = self.result.layers[n_pen] if n_pen < len(self.result.layers) else None
            df   = st.damage_fraction if st else 0.5
            return rect.left + int(rect.width * df)
        return self.layer_rects[-1].right

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self):
        self.frame += 1
        if self.frame >= TOTAL_FRAMES:
            self.done = True

        # Update effects
        for p in self.particles: p.update()
        for r in self.rings:     r.update()
        for b in self.blooms:    b.update()
        self.particles = [p for p in self.particles if p.life > 0]
        self.rings     = [r for r in self.rings     if r.alive]
        self.blooms    = [b for b in self.blooms    if b.alive]

        if self._breach_flash > 0:
            self._breach_flash -= 1

        # Trigger layer events — only for layers the weapon actually reaches
        if self.penetrate_t > 0 and self.layer_rects:
            for i in range(self._n_active):
                if i >= len(self._layer_trigger_t):
                    break
                if self._layer_triggered[i]:
                    continue
                if self.penetrate_t >= self._layer_trigger_t[i]:
                    self._layer_triggered[i] = True
                    self._spawn_layer_event(i)

    def _spawn_layer_event(self, idx):
        if idx >= len(self.layer_rects):
            return
        rect  = self.layer_rects[idx]
        wtype = self.weapon_type
        color = WEAPON_COLORS.get(wtype, (200, 200, 200))
        shock = SHOCK_COLORS.get(wtype, (255, 255, 200))
        cy    = rect.centery
        x     = rect.left + 4

        st     = self.result.layers[idx] if (self.result and idx < len(self.result.layers)) else None
        temp   = st.temperature_k if st else 300.0
        pen    = st.penetrated    if st else False

        # Temperature-based spark color
        if temp > 3000:
            spark_c = (255, 255, 220)
        elif temp > 1500:
            spark_c = (255, 180,  60)
        elif temp > 800:
            spark_c = (255, 100,  30)
        else:
            spark_c = color

        # ── Shockwave rings ──
        n_rings = 3 if pen else 5
        for k in range(n_rings):
            r_max = int(rect.height * (0.4 + k * 0.25))
            life  = 22 + k * 8
            self.rings.append(ShockRing(x, cy, r_max, life, shock, width=max(1, 3 - k)))

        # ── Heat bloom ──
        bloom_rw = rect.width if pen else int(rect.width * (st.damage_fraction if st else 0.5))
        bloom_rh = int(rect.height * 0.7)
        self.blooms.append(HeatBloom(x, cy, bloom_rw, bloom_rh, 35, spark_c))

        # ── Spall / debris particles ──
        n_sparks = 18 if pen else 30
        if wtype == 'explosive':
            n_sparks = 60
        elif wtype == 'laser':
            n_sparks = 12

        for _ in range(n_sparks):
            if pen:
                angle = random.uniform(-math.pi / 3, math.pi / 3)
                speed = random.uniform(2.0, 8.0)
            else:
                angle = random.uniform(-math.pi * 0.7, math.pi * 0.7)
                speed = random.uniform(1.0, 5.0)

            if wtype == 'explosive':
                angle = random.uniform(-math.pi, math.pi)
                speed = random.uniform(2.0, 12.0)
                x     = rect.centerx

            vx   = math.cos(angle) * speed
            vy   = math.sin(angle) * speed - random.uniform(0, 1.5)
            life = random.randint(20, 50)
            sz   = random.randint(1, 4 if pen else 3)
            self.particles.append(Particle(x, cy, vx, vy, life, spark_c, size=sz))

        # ── Crack lines (static line particles for stopped layers) ──
        if not pen and st and st.damage_fraction > 0.1:
            for _ in range(6):
                ang = random.uniform(-math.pi / 2, math.pi / 2)
                length = random.randint(8, int(rect.width * st.damage_fraction * 0.8 + 4))
                ex = x + int(math.cos(ang) * length)
                ey = cy + int(math.sin(ang) * length)
                # Represent as very slow, zero-gravity particle pair
                self.particles.append(Particle(x, cy, 0, 0, 40,
                                               (180, 80, 40), size=1, gravity=0))
                self.particles.append(Particle(ex, ey, 0, 0, 40,
                                               (180, 80, 40), size=1, gravity=0))

        # ── Breach flash ──
        if pen and self.result and self.result.breached and idx == self.result.layers_penetrated - 1:
            self._breach_flash = 10

            # Exit spray from back face
            back_x = self.layer_rects[-1].right
            for _ in range(40):
                angle = random.uniform(-math.pi / 2.5, math.pi / 2.5)
                speed = random.uniform(3.0, 11.0)
                vx    = math.cos(angle) * speed
                vy    = math.sin(angle) * speed - random.uniform(0, 2)
                life  = random.randint(25, 60)
                self.particles.append(Particle(back_x, cy, vx, vy, life,
                                               spark_c, size=random.randint(2, 5)))

    # ── Render ────────────────────────────────────────────────────────────────

    def render_overlay(self, surface, offset_x=0, offset_y=0):
        if not self.layer_rects:
            return

        wtype    = self.weapon_type
        color    = WEAPON_COLORS.get(wtype, (200, 200, 200))
        first    = self.layer_rects[0]
        cy       = first.centery + offset_y
        wx       = self._weapon_x() + offset_x

        # ── Heat blooms (drawn first, behind everything) ──────────
        for b in self.blooms:
            b.draw(surface)

        # ── Shock rings ───────────────────────────────────────────
        for ring in self.rings:
            ring.draw(surface)

        # ── Weapon body / beam ────────────────────────────────────
        phase = self.frame

        if phase < APPROACH_END or (phase < PENETRATE_END and wtype not in ('laser', 'particle')):
            if wtype == 'kinetic':
                _draw_projectile(surface, wx, cy + offset_y, color)
            elif wtype == 'laser':
                _draw_laser_beam(surface, self.weapon_zone_x + offset_x,
                                 cy + offset_y, wx, color,
                                 pulse=self.frame % 4 < 2)
            elif wtype == 'plasma':
                _draw_plasma_front(surface, wx, cy + offset_y, first.height, color)
            elif wtype == 'explosive':
                _draw_shockwave(surface, wx, cy + offset_y,
                                self.approach_t, first.height, color)
            elif wtype == 'particle':
                _draw_beam_trail(surface, self.weapon_zone_x + offset_x,
                                 cy + offset_y, wx, color)
            elif wtype == 'gravitational':
                _draw_gravity_ripple(surface, wx, cy + offset_y,
                                     first.height, self.approach_t, color)

        # Laser/particle continue into armor during penetration
        elif wtype in ('laser', 'particle') and phase < PENETRATE_END:
            if wtype == 'laser':
                _draw_laser_beam(surface, self.weapon_zone_x + offset_x,
                                 cy + offset_y, wx, color,
                                 pulse=self.frame % 4 < 2)
            else:
                _draw_beam_trail(surface, self.weapon_zone_x + offset_x,
                                 cy + offset_y, wx, color)

        # Kinetic projectile continues through penetrate phase
        elif wtype == 'kinetic' and phase < SETTLE_END:
            _draw_projectile(surface, wx, cy + offset_y, color)

        # ── Breach flash ──────────────────────────────────────────
        if self._breach_flash > 0 and self.result and self.result.breached:
            alpha = int(220 * self._breach_flash / 10)
            flash = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            flash.fill((255, 120, 40, alpha))
            surface.blit(flash, (0, 0))

        # ── Particles ─────────────────────────────────────────────
        for p in self.particles:
            p.draw(surface)

        # ── Slow-mo label ─────────────────────────────────────────
        if phase < SETTLE_END:
            try:
                f = pygame.font.SysFont('consolas', 10)
                lbl = f.render('SLOW  MOTION', True, (60, 80, 100))
                surface.blit(lbl, (surface.get_width() - lbl.get_width() - 6,
                                   surface.get_height() - lbl.get_height() - 4))
            except Exception:
                pass


# ── Draw helpers ─────────────────────────────────────────────────────────────

def _draw_projectile(surface, x, cy, color):
    # Elongated rod with motion-blur ghost trail
    for i in range(5):
        alpha = 1.0 - i * 0.2
        c = tuple(int(v * alpha) for v in color)
        pygame.draw.rect(surface, c, (x - 20 - i * 7, cy - 4, 20, 8))
    # Nose cap
    pygame.draw.ellipse(surface, color, (x - 20, cy - 5, 22, 10))
    # Bright leading edge
    pygame.draw.circle(surface, (255, 255, 200), (x, cy), 3)


def _draw_laser_beam(surface, x0, cy, x1, color, pulse=False):
    if x1 <= x0:
        return
    c = color if not pulse else tuple(min(255, int(v * 1.3)) for v in color)
    pygame.draw.line(surface, c, (x0, cy), (x1, cy), 2)
    for w, alpha in [(6, 50), (12, 25), (20, 10)]:
        s = pygame.Surface((x1 - x0, w), pygame.SRCALPHA)
        s.fill((*color, alpha))
        surface.blit(s, (x0, cy - w // 2))
    # Hot tip
    pygame.draw.circle(surface, (255, 255, 255), (x1, cy), 3)


def _draw_plasma_front(surface, x, cy, height, color):
    h = int(height * 0.55)
    for r, alpha in [(h, 12), (h // 2, 28), (h // 4, 60)]:
        if r < 1:
            continue
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*color, alpha), (0, 0, r * 2, r * 2))
        surface.blit(s, (x - r, cy - r))
    pygame.draw.ellipse(surface, color, (x - 10, cy - 14, 20, 28))
    pygame.draw.circle(surface, (255, 240, 200), (x, cy), 4)


def _draw_shockwave(surface, x, cy, t, height, color):
    r_max = int(height * 0.65)
    r = int(r_max * (0.25 + 0.75 * t))
    for offset, alpha in [(0, 70), (10, 45), (22, 20)]:
        rr = r + offset
        if rr < 1:
            continue
        s = pygame.Surface((rr * 2 + 2, rr * 2 + 2), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*color, alpha), (0, 0, rr * 2, rr * 2), 3)
        surface.blit(s, (x - rr, cy - rr))
    pygame.draw.circle(surface, color, (x, cy), max(2, r // 5))


def _draw_beam_trail(surface, x0, cy, x1, color):
    if x1 <= x0:
        return
    seg = 7
    for sx in range(x0, x1, seg * 2):
        ex = min(sx + seg, x1)
        pygame.draw.line(surface, color, (sx, cy), (ex, cy), 1)
    pygame.draw.circle(surface, (220, 255, 220), (x1, cy), 3)


def _draw_gravity_ripple(surface, x, cy, height, t, color):
    h = height // 2
    for i in range(4):
        phase = i * (math.pi / 2) + t * 7
        for dy in range(-h, h, 3):
            wave_x = x + int(14 * math.sin(phase + dy * 0.06))
            pygame.draw.circle(surface, color, (wave_x, cy + dy), 1)
