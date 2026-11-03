# Daphnis and the Keeler Gap waves: what the picture is, what it will look like, and who can see it (2026-11-02)

The Captain's ask: that image (Cassini, Daphnis in the Keeler Gap with scalloped edges and long shadows) is mathematically important; explain it fully, predict what it will look like in the future, and say whether it will be visible. Every number below is sourced or is Mentat's arithmetic on sourced inputs; the open points are marked.

## 1. The inputs (sourced)
| quantity | value | source |
|---|---|---|
| Daphnis mass | (6.8 ± 1.7) e13 kg | Thomas & Helfenstein 2020, Icarus (via the Wikipedia article's citation) |
| Daphnis size | 9.8 x 8.4 x 5.6 km, mean diameter 7.8 ± 1.0 km | same |
| semi-major axis | 136,505.5 ± 0.1 km | Jacobson et al. 2008, AJ, "Revised orbits of Saturn's small inner satellites" |
| eccentricity | 0.0000331 ± 0.0000062 | same |
| inclination | 0.0036 ± 0.0013 deg | same |
| period | 0.5940798 d (14.258 h) | same; JPL SSD elements page gives 0.594080 d (SAT415) |
| Keeler Gap width | 42 km | NASA/JPL caption of PIA11654 |
| vertical structure height (2009) | 0.5 to 1.5 km, image of 8 June 2009, phase 130 deg, 48 deg above the ring plane | NASA/JPL caption of PIA11654 ("Rippling Shadows") |
| Saturn GM / mass | 37,931,206.234 km^3/s^2; 5.683e26 kg | Mentat `body_constants('saturn')` (SAT441) |
| the theory | edge waves from an embedded moon; the simple analytic theory fits Pan in the Encke Gap and BREAKS DOWN for Daphnis (edge within a few Hill radii): analytic mass estimates come out too high; a fitted model from test-particle simulations replaces it; inclined orbits give the waves a vertical component; shadows are ~1 km structures | Weiss, Porco & Tiscareno 2009, AJ 138, 272 (abstract; DOI 10.1088/0004-6256/138/1/272; full text not read) |

Mentat has no entry for Daphnis (`body_constants('daphnis')` declines: the catalogue stops at the 40-odd larger moons). Adding the ring-region moons (Pan, Daphnis, Atlas, Prometheus, Pandora, Janus, Epimethus) with SAT415 masses is a small data task recorded in the campaign brief.

## 2. What the picture is (the mechanism)
* Daphnis orbits inside the gap. Ring particles just inside its orbit overtake it (they orbit faster), those just outside fall behind. Each particle gets one gravitational tug as it passes, which turns its circular orbit into a slightly eccentric one. Because all the particles at one radius get the same kick at the same place relative to the moon, their eccentric orbits line up into a standing wave in the moon's frame: a wavy edge that travels with Daphnis.
* Wavelength: one epicycle per synodic period, which for a particle at radial separation da from the moon is lambda = 3 pi da (Keplerian shear). With da = 21 km (half the gap): **lambda = 198 km** (Mentat). The scallops in the image are about that spacing; the exact da differs between the inner and outer edge and with Daphnis's eccentric excursion (a e = 4.5 km), so the two edges' wavelengths differ and change along the orbit.
* Radial amplitude (impulse approximation, A = 2.24 (m/M) (a/da)^2 a): **1.55 km** (Mentat, with the 2020 mass). Weiss et al. say this simple theory is the one that fails for Daphnis (too close to the edge in Hill radii), so this is an order of magnitude, not a prediction; their simulation-fitted relation (full text not read) is what a real prediction needs.
* Vertical waves: Daphnis's orbit is tilted 0.0036 deg, so it rises **8.6 km** above and below the ring plane each orbit (Mentat: a sin i). Particles it passes while it is above the plane get an upward kick, and their vertical oscillation (period = the orbital period, nearly) stays in phase in the moon's frame, so the edge gets a vertical ripple as well as a radial one. Cassini's shadows measured it at **0.5 to 1.5 km**, a fraction of the moon's own excursion, as expected for a kick from several Hill radii away.
* Damping: the wave is visible for several wavelengths and then the edge is smooth again. Ring particles collide about once an orbit; the collisions convert the organised epicyclic and vertical motion into heat and re-circularise the orbits. (Weiss et al. model this with test particles, no collisions, which is why their model is a lower bound on the damping; the real ring damps within a few lambda, as the image shows.) The plane the ring relaxes to is set by Saturn's oblateness (J2 = 0.016290573, Cassini Grand Finale): the equatorial plane is the Laplace plane this close to the planet. No magnetic or relativistic effect enters: the particles are neutral ice and the field is Newtonian to far better than the picture resolves.

## 3. What it will look like in the future (predictions, written before any new image exists)
1. **The pattern is permanent.** It is stationary in Daphnis's frame and renewed every orbit (14.26 h). It will look the same, with the same ~200 km wavelength and ~1-2 km radial amplitude, for as long as Daphnis stays in the gap. The moon's orbit decays only through the same ring torques that made the gap, on timescales of millions of years or more. Nothing changes on a human timescale.
2. **The vertical part has a season.** The vertical kick depends on Daphnis's height when it passes each particle, which depends on where its ascending node is relative to the moon-particle conjunction. The node precesses under Saturn's J2 (period of order a year for an orbit this close; the exact rate is a `perihelion_advance`-type calculation Mentat can do once Daphnis is in the catalogue), so the vertical wave's phase and amplitude pattern cycle with the nodal period. The 0.5-1.5 km shadows of June 2009 were one phase of that cycle.
3. **Shadows need equinox.** The vertical ripple is 1 km on a ring 40,000 km wide; the only way to see it is the shadow it casts when the Sun is within about a degree of the ring plane. Saturn's equinoxes: 11 Aug 2009 (Cassini saw it), 6 May 2025 (no spacecraft at Saturn), next about **2040.1** (Mentat: 2025.35 + half of 29.457 yr), then ~2054.8.
4. **The radial wave can be imaged at any time** by a camera at Saturn with better than ~50 km resolution.

## 4. Who can see it
* **From Earth or Earth orbit: no.** JWST's diffraction limit at 2 um is 0.077 arcsec (Mentat: 1.22 lambda/D), which is **477 km at Saturn's distance** (8.5 au): ten times the gap's width and twice the wavelength of the waves. Hubble and the ground are worse. The Keeler Gap itself is below their resolution; the waves are far below it.
* **Cassini is gone** (plunged 15 Sep 2017). No Saturn orbiter is approved as of this writing; Dragonfly (launch 2028, arrival 2034) is a Titan lander with no ring imaging. So **the 2040 equinox shadows will go unobserved unless an orbiter is sent by then**; the radial waves likewise.
* Prediction for the record: the next image of this structure will come from a spacecraft, not a telescope, and not before the 2030s.

## 5. What would make this a Mentat capability (not done here)
A `ring_edge_wave(moon, gap_half_width_km)` tool: wavelength 3 pi da, the impulse amplitude with the Weiss et al. correction factor once the paper's fitted relation is read, the vertical amplitude from the inclination, and a `next_equinox('saturn')` from the planet's orbit. Inputs need the ring-region moons in the body catalogue (SAT415 masses). Listed in misc/NBODY_CAMPAIGN.md.

Mentat defects met here (for the ledger): `evaluate_quantity` has no `sin`; "2.0 micrometer / 6.5 meter" was parsed as micrometer x meter (pint's division precedence: write the ratio in parentheses).
