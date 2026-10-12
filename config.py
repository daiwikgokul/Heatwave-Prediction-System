"""Central configuration: cities, regions, IMD thresholds, palettes, paths."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODELS = BASE / "models"
ASSETS = BASE / "assets"

DATA_START = "2005-01-01"
DATA_END = "2026-10-06"
TRAIN_END = "2022-12-31"      # models are fitted on data up to here
VAL_END = "2023-12-31"        # 2023 = validation (interval calibration, threshold tuning)
# 2024-01-01 onwards = hold-out test / "operational" period
AWS_START = "2025-01-01"      # IoT stations "installed" on this date
HORIZONS = list(range(1, 8))  # forecast lead times in days

SEASONS = {  # IMD four climatological seasons
    "Winter": [1, 2],
    "Pre-monsoon": [3, 4, 5],
    "Monsoon": [6, 7, 8, 9],
    "Post-monsoon": [10, 11, 12],
}
MONTH_TO_SEASON = {m: s for s, ms in SEASONS.items() for m in ms}

# ---- IMD heatwave criteria (plains / coastal) -------------------------------
IMD = {
    "plains": {"base": 40.0, "abs_hw": 45.0, "abs_severe": 47.0},
    "coastal": {"base": 37.0, "abs_hw": None, "abs_severe": None},
}
DEP_HOT = 3.0      # departure from normal for a "hot day" (watch)
DEP_HW = 4.5       # departure for heatwave
DEP_SEVERE = 6.5   # departure for severe heatwave

SEVERITY = ["Normal", "Hot Day", "Heatwave", "Severe Heatwave"]
ALERT_LEVELS = {
    0: {"colour": "Green", "name": "No warning", "hex": "#2E9E5B", "action": "Normal activity"},
    1: {"colour": "Yellow", "name": "Watch", "hex": "#F2C230", "action": "Be updated"},
    2: {"colour": "Orange", "name": "Alert", "hex": "#F28C28", "action": "Be prepared"},
    3: {"colour": "Red", "name": "Warning", "hex": "#D62839", "action": "Take action"},
}
SEV_COLOURS = [ALERT_LEVELS[i]["hex"] for i in range(4)]

# Heatwave Risk Index weights on class probabilities and alert-trigger rules
HRI_WEIGHTS = [0.0, 0.25, 0.65, 1.0]
# Alert tiers: highest tier whose rule is met on a forecast day. Thresholds were read off the
# CSI-vs-threshold curves on the 2023 validation year (flat optimum around 0.12-0.25) and rounded.
ALERT_RULES = {"red_p_severe": 0.15, "orange_p_hw": 0.20, "yellow_p_any": 0.25}

# ---- palette ----------------------------------------------------------------
INK = "#12263A"
SLATE = "#5B6B7C"
PAPER = "#F1F4F8"
SURFACE = "#FFFFFF"
BLUE = "#1F4E8C"
THERMAL = ["#3B6FB6", "#8FC3D9", "#F6E8A6", "#F6B35A", "#E4572E", "#8E1B2E"]

REGIONS = [
    "Northwest India", "West India", "North-Central (Indo-Gangetic)", "Central India",
    "Deccan Plateau", "South Peninsula (East Coast)", "East India",
]

# monthly climatological means (synthetic, loosely based on published normals)
CITIES = [
    dict(name="Delhi", code="DEL", state="Delhi", region="Northwest India", lat=28.61, lon=77.21, elev=216,
         terrain="plains", pop=32.0, suscept=1.00, sig=1.00, dtr=1.00, wind=9,
         tmax=[21.3, 24.5, 30.4, 36.5, 39.6, 38.9, 35.0, 33.7, 34.1, 33.0, 28.3, 23.2],
         rh=[38, 30, 24, 17, 20, 35, 62, 68, 55, 35, 34, 40]),
    dict(name="Jaipur", code="JAI", state="Rajasthan", region="Northwest India", lat=26.91, lon=75.79, elev=431,
         terrain="plains", pop=4.2, suscept=1.05, sig=1.00, dtr=1.05, wind=11,
         tmax=[22.5, 25.5, 31.5, 37.5, 41.0, 39.5, 34.5, 32.5, 33.5, 33.5, 29.0, 24.0],
         rh=[32, 26, 20, 15, 18, 34, 60, 66, 50, 28, 28, 32]),
    dict(name="Ahmedabad", code="AMD", state="Gujarat", region="West India", lat=23.02, lon=72.57, elev=55,
         terrain="plains", pop=8.4, suscept=1.00, sig=0.90, dtr=0.95, wind=10,
         tmax=[29.0, 31.5, 36.0, 39.5, 41.5, 38.5, 33.5, 32.0, 33.5, 35.5, 33.0, 30.0],
         rh=[28, 22, 18, 20, 30, 48, 66, 70, 56, 32, 28, 28]),
    dict(name="Lucknow", code="LKO", state="Uttar Pradesh", region="North-Central (Indo-Gangetic)", lat=26.85, lon=80.95,
         elev=123, terrain="plains", pop=4.0, suscept=1.00, sig=1.05, dtr=1.00, wind=8,
         tmax=[22.0, 25.5, 31.5, 37.5, 40.0, 38.5, 34.0, 33.0, 33.5, 33.0, 28.5, 23.5],
         rh=[42, 32, 25, 18, 22, 38, 65, 70, 58, 38, 38, 44]),
    dict(name="Nagpur", code="NAG", state="Maharashtra", region="Central India", lat=21.15, lon=79.09, elev=310,
         terrain="plains", pop=3.0, suscept=0.90, sig=0.90, dtr=0.95, wind=9,
         tmax=[29.5, 32.5, 36.5, 40.5, 42.0, 36.5, 31.5, 31.0, 32.0, 33.0, 30.5, 28.5],
         rh=[30, 24, 19, 16, 18, 52, 68, 70, 58, 38, 32, 32]),
    dict(name="Mumbai", code="BOM", state="Maharashtra", region="West India", lat=19.08, lon=72.88, elev=14,
         terrain="coastal", pop=21.0, suscept=0.45, sig=0.50, dtr=0.50, wind=12,
         tmax=[31.5, 32.0, 33.0, 33.5, 33.5, 32.0, 29.5, 29.5, 30.5, 33.0, 34.0, 33.0],
         rh=[50, 52, 56, 60, 62, 75, 85, 85, 78, 62, 52, 50]),
    dict(name="Pune", code="PNQ", state="Maharashtra", region="Deccan Plateau", lat=18.52, lon=73.86, elev=560,
         terrain="plains", pop=7.0, suscept=0.85, sig=0.80, dtr=0.95, wind=9,
         tmax=[30.0, 32.5, 35.5, 37.5, 36.0, 30.5, 27.5, 27.5, 29.0, 31.5, 30.5, 29.5],
         rh=[30, 24, 22, 24, 35, 64, 76, 76, 64, 46, 38, 32]),
    dict(name="Hyderabad", code="HYD", state="Telangana", region="Deccan Plateau", lat=17.39, lon=78.49, elev=542,
         terrain="plains", pop=11.0, suscept=0.95, sig=0.75, dtr=0.85, wind=10,
         tmax=[29.5, 32.5, 35.5, 38.0, 39.0, 33.5, 30.5, 30.0, 30.5, 30.5, 29.0, 28.5],
         rh=[35, 27, 22, 24, 28, 56, 64, 64, 60, 50, 42, 38]),
    dict(name="Chennai", code="MAA", state="Tamil Nadu", region="South Peninsula (East Coast)", lat=13.08, lon=80.27,
         elev=6, terrain="coastal", pop=12.0, suscept=0.55, sig=0.55, dtr=0.55, wind=13,
         tmax=[29.0, 30.5, 32.5, 35.0, 38.0, 37.5, 36.0, 35.0, 34.5, 32.0, 29.5, 28.5],
         rh=[62, 60, 58, 58, 48, 48, 50, 54, 58, 66, 70, 66]),
    dict(name="Bhubaneswar", code="BBI", state="Odisha", region="East India", lat=20.30, lon=85.82, elev=45,
         terrain="plains", pop=1.2, suscept=0.90, sig=0.70, dtr=0.70, wind=8,
         tmax=[29.0, 32.0, 35.5, 37.5, 37.5, 34.5, 32.0, 31.5, 32.0, 31.5, 29.5, 28.0],
         rh=[50, 46, 44, 50, 54, 68, 76, 76, 74, 64, 52, 50]),
]
CITY_NAMES = [c["name"] for c in CITIES]
CITY = {c["name"]: c for c in CITIES}
CITY_ID = {c["name"]: i for i, c in enumerate(CITIES)}

# base diurnal temperature range by month for an inland plains city (deg C)
DTR_MONTHLY = [14.0, 15.0, 15.0, 14.5, 12.5, 9.5, 6.5, 6.0, 7.5, 11.0, 13.5, 14.0]
# monthly std-dev of daily Tmax anomaly (before city scaling)
SIGMA_MONTHLY = [2.2, 2.3, 2.3, 2.0, 1.9, 1.9, 1.2, 1.0, 1.1, 1.3, 1.5, 2.0]

# ---- AWS (IoT) stations: 4 per city ----------------------------------------
# (type -> tmax offset degC, RH offset %, description)
STATION_TYPES = {
    "Urban Core": (1.3, -5.0, "Dense built-up area, low vegetation"),
    "Industrial": (0.9, -4.0, "Industrial estate, heat-emitting surfaces"),
    "Suburban": (0.2, 0.0, "Residential, mixed land cover"),
    "Green Belt": (-0.8, 5.0, "Park, lake or forest edge"),
}
AWS_SITES = {
    "Delhi": [("Connaught Place", "Urban Core", 28.633, 77.219), ("Okhla", "Industrial", 28.530, 77.270),
              ("Dwarka", "Suburban", 28.590, 77.050), ("Aravalli Ridge", "Green Belt", 28.680, 77.170)],
    "Jaipur": [("Johari Bazaar", "Urban Core", 26.918, 75.826), ("Sitapura", "Industrial", 26.780, 75.850),
               ("Mansarovar", "Suburban", 26.850, 75.760), ("Nahargarh Foothills", "Green Belt", 26.940, 75.810)],
    "Ahmedabad": [("Lal Darwaja", "Urban Core", 23.025, 72.580), ("Naroda GIDC", "Industrial", 23.070, 72.650),
                  ("Thaltej", "Suburban", 23.050, 72.500), ("Kankaria Lake", "Green Belt", 23.007, 72.600)],
    "Lucknow": [("Hazratganj", "Urban Core", 26.850, 80.940), ("Amausi", "Industrial", 26.760, 80.880),
                ("Gomti Nagar", "Suburban", 26.850, 81.000), ("Janeshwar Mishra Park", "Green Belt", 26.860, 81.020)],
    "Nagpur": [("Sitabuldi", "Urban Core", 21.145, 79.085), ("Hingna MIDC", "Industrial", 21.100, 78.980),
               ("Dharampeth", "Suburban", 21.140, 79.060), ("Ambazari Lake", "Green Belt", 21.130, 79.040)],
    "Mumbai": [("Fort", "Urban Core", 18.935, 72.835), ("Chembur", "Industrial", 19.060, 72.900),
               ("Powai", "Suburban", 19.120, 72.905), ("Borivali National Park", "Green Belt", 19.210, 72.910)],
    "Pune": [("Shivajinagar", "Urban Core", 18.530, 73.850), ("Pimpri-Chinchwad", "Industrial", 18.630, 73.800),
             ("Hadapsar", "Suburban", 18.500, 73.930), ("Vetal Tekdi", "Green Belt", 18.520, 73.820)],
    "Hyderabad": [("Charminar", "Urban Core", 17.360, 78.475), ("Patancheru", "Industrial", 17.530, 78.265),
                  ("Gachibowli", "Suburban", 17.440, 78.350), ("KBR Park", "Green Belt", 17.420, 78.430)],
    "Chennai": [("T. Nagar", "Urban Core", 13.040, 80.230), ("Ennore", "Industrial", 13.215, 80.320),
                ("Velachery", "Suburban", 12.980, 80.220), ("Guindy National Park", "Green Belt", 13.005, 80.235)],
    "Bhubaneswar": [("Old Town", "Urban Core", 20.240, 85.840), ("Mancheswar", "Industrial", 20.300, 85.860),
                    ("Patia", "Suburban", 20.350, 85.820), ("Nandankanan", "Green Belt", 20.395, 85.820)],
}
