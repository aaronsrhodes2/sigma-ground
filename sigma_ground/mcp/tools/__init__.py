"""MCP tool implementations.

Each module exposes a small set of related physics calculations as
plain functions returning ToolResult. server.py registers them with
the MCP server, exposing them to the LLM front-end.

Modules:
    constants      CODATA + sigma_ground tagged constants
    units          pint-based unit conversion
    symbolic       sympy solve/integrate/simplify
    gr             general-relativity calculations
    electrodynamics  Coulomb, Bohr, atomic physics
    cosmology      Hubble, HDE, MOND
    nbody          rolling-shootout-style predictions
"""
