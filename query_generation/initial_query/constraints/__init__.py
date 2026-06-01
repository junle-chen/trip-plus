"""Constraint helpers for initial query generation.

The package keeps the initial-query constraint flow split by role:

- ``hard`` selects and builds database-verifiable hard constraints.
- ``transport``, ``lodging``, ``dining``, and ``attractions`` build
  database-verifiable category constraints.
- ``transport_routes`` samples feasible train/flight route frames before
  record construction.
- ``route_manifests`` stores bundled seasonal route lists used by
  ``transport_routes``.
- ``explicit_trip`` promotes trip-frame metadata such as dates, party size,
  room count, and intercity mode into evaluator-readable hard constraints.
- ``evidence`` contains shared row filters, option payloads, and label cleanup.
- ``budget`` adjusts the optional hard budget constraint.
- ``environment`` derives city, weather, and practical environment hints.
"""
