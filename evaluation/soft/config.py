"""Soft-preference rule ids, families, and marker vocabularies."""

SOFT_CHECK_RULE_IDS = {
    "schedule_pacing",
    "mobility_accessibility",
    "weather_avoid_heat_exposure",
    "weather_avoid_cold_exposure",
    "weather_need_backup",
    "transport_avoid_transfer",
    "transport_avoid_red_eye",
    "transport_avoid_early_departure",
    "transport_avoid_late_arrival",
    "transport_prefer_train",
    "transport_prefer_flight",
    "hotel_value_first",
    "budget_guarded",
    "budget_tight_cap",
    "meal_avoid_expensive",
    "interest_local_food",
    "interest_outdoor_nature",
    "interest_culture",
    "interest_art",
    "interest_shopping",
    "interest_landmark",
    "interest_amusement",
}

SOFT_PREFERENCE_FAMILIES = {
    "comfort_and_pace": {
        "schedule_pacing",
        "mobility_accessibility",
        "weather_avoid_heat_exposure",
        "weather_avoid_cold_exposure",
        "weather_need_backup",
    },
    "transport_convenience": {
        "transport_avoid_transfer",
        "transport_avoid_red_eye",
        "transport_avoid_early_departure",
        "transport_avoid_late_arrival",
        "transport_prefer_train",
        "transport_prefer_flight",
    },
    "budget_and_value": {
        "hotel_value_first",
        "budget_guarded",
        "budget_tight_cap",
        "meal_avoid_expensive",
    },
    "interest_match": {
        "interest_local_food",
        "interest_outdoor_nature",
        "interest_culture",
        "interest_art",
        "interest_shopping",
        "interest_landmark",
        "interest_amusement",
    },
}

SOFT_PREFERENCE_FAMILY_ORDER = (
    "comfort_and_pace",
    "transport_convenience",
    "budget_and_value",
    "interest_match",
)

OUTDOOR_MARKERS = (
    "mountain",
    "canyon",
    "lake",
    "bay",
    "island",
    "sea",
    "beach",
    "park",
    "forest",
    "grassland",
    "wetland",
    "ancient town",
    "street",
    "temple",
    "trail",
    "boardwalk",
)
REST_MARKERS = ("rest", "break", "buffer", "free time", "hotel")
NATURE_MARKERS = (
    "mountain",
    "lake",
    "bay",
    "sea",
    "beach",
    "park",
    "forest",
    "grassland",
    "wetland",
    "canyon",
    "nature",
    "natural",
)
PARK_MARKERS = ("park", "wetland", "garden")
HISTORY_MARKERS = (
    "ancient",
    "history",
    "historic",
    "historical",
    "museum",
    "ruins",
    "memorial",
    "culture",
    "cultural",
    "temple",
    "city wall",
)
MUSEUM_MARKERS = ("museum", "memorial", "exhibition", "gallery", "science")
ART_MARKERS = ("art", "gallery", "exhibition")
SHOPPING_MARKERS = (
    "mall",
    "pedestrian street",
    "commercial",
    "shopping",
    "square",
    "market",
)
LANDMARK_MARKERS = (
    "tower",
    "building",
    "square",
    "landmark",
    "center",
    "centre",
    "city wall",
    "ancient city",
)
AMUSEMENT_MARKERS = ("amusement", "theme park", "ocean park", "aquarium", "zoo")
FOOD_MARKERS = (
    "meal",
    "food",
    "restaurant",
    "cuisine",
    "snack",
    "local",
    "hot pot",
    "noodle",
    "rice noodle",
    "dining",
)
INDOOR_MARKERS = (
    "museum",
    "memorial",
    "gallery",
    "exhibition",
    "exhibit",
    "mall",
    "shopping center",
    "shopping centre",
    "indoor",
    "theater",
    "theatre",
    "science",
    "library",
    "aquarium",
    "art center",
    "art centre",
)
LOCAL_FOOD_DB_MARKERS = (
    "local",
    "specialty",
    "traditional",
    "heritage",
    "snack",
    "street food",
    "signature",
    "regional",
    "time-honored",
    "time honoured",
    "old brand",
    "noodle",
    "hot pot",
)
DB_ATTRACTION_FIELDS = ("attraction_type", "description", "popularity_tags")
DB_RESTAURANT_FIELDS = ("cuisine", "tags")

SEVERITY_SCORES = {"none": 1.0, "minor": 0.5, "major": 0.0}
SEVERITY_RANK = {"none": 0, "minor": 1, "major": 2}
