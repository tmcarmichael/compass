"""Navigation: terrain interpretation, pathfinding, and movement control.

Terrain:
    terrain/heightmap.py  -  1-unit-resolution grid from zone geometry
                             walkable/water/lava/cliff/obstacle flags, material
                             classification, bridge handling, avoidance zones

Pathfinding:
    pathfinding.py        -  JPS/A* over the terrain grid with surface-aware costs,
                             hazard inflation, path variation for repeated traversals
    zone_graph.py         -  Inter-zone BFS routing from map POI data
    waypoint_graph.py     -  Indoor/complex-geometry waypoint routing
    travel_planner.py     -  Multi-leg route planning (A* + manual waypoint legs)

Movement:
    movement.py           -  Closed-loop move_to_point with heading control,
                             cancellation, and escalating stuck recovery
    stuck.py              -  Displacement-based stuck detector

Support:
    geometry.py           -  Heading, distance, angle, facing calculations
    map_data.py           -  Map segment parsing for obstacle queries
"""

__all__: list[str] = []
