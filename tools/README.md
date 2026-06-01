# Tools

`tools/` implements the travel lookup tools exposed to the planner. Tools read
fixed benchmark databases and return compact evidence for flights, trains,
hotels, restaurants, attractions, weather, local transport, and location search.

The default multi-turn release uses the same tools and sample databases as the
full pipeline. Tools do not read query split views such as `query/.../items/`;
they receive resolved sample metadata from the runner.

## Files

- `base_travel_tool.py`: shared tool base class and parameter validation.
- `sample_db_resolver.py`: maps query IDs to per-sample database directories.
- `city_db_access.py`: city/sample database resolution and CSV/JSON reads.
- `tool_schema_en.json`: function schema exposed to the agent.
- `flight_query_tool.py`: flight lookup.
- `train_query_tool.py`: train lookup.
- `hotel_query_tool.py`: hotel lookup.
- `restaurant_query_tool.py`: restaurant lookup.
- `attraction_query_tool.py`: attraction lookup.
- `weather_query_tool.py`: city weather lookup.
- `city_transport_query_tool.py`: local transport lookup.
- `roadroute_query_tool.py`: coordinate-level road route lookup.
- `location_search_tool.py`: location search.
- `weather_utils.py`: shared weather formatting helpers.

## Boundary

- Tools answer database-backed lookup questions.
- Planning strategy belongs in `agent/`.
- Scoring logic belongs in `evaluation/`.
