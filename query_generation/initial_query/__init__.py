"""English single-turn query construction.

Read the package in this order:

1. ``cli.py`` parses command-line options and writes outputs.
2. ``pipeline.py`` runs the sampling loop.
3. ``records.py`` assembles one benchmark record from a route, profile, and city DB.
4. ``rendering.py`` turns the structured record into the visible user query.
5. ``visibility`` repairs the rendered text so visible hard constraints stay evaluable.
"""
