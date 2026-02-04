# app.py
# ============================================================
# Soil Moisture Sensor Optimization – Gradio App
# ------------------------------------------------------------
# - Upload field GeoJSON (or draw AOI on map & export)
# - For a given date:
#       • Build grids at various cell sizes
#       • Predict SM (ExtraTrees) at centroids
#       • Compute CV (%)
#       • Return optimal N sensors (min CV)
# - UI:
#       • Left: inputs + AOI drawer (folium Draw + search box)
#       • Right tabs:
#           - Optimization (CV vs N + table)
#           - Sensor layout (centroid map + coordinates)
#
# EE auth: via env vars
#   EE_SERVICE_ACCOUNT       – service account email
#   EE_PROJECT_ID            – EE project id
#   EE_SERVICE_ACCOUNT_KEY   – full SA JSON as a single string
#
# Model files expected in repo root:
#   extratrees_s1_soil_moisture_points.pkl
#   extratrees_s1_soil_moisture_features.txt
#
# Example AOI:
#   examples/example_field.geojson
# ============================================================

import os
import json
import math
import requests

import ee
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

import folium
from folium.plugins import Draw

# ------------------------------------------------------------
# Paths for model + example AOI
# ------------------------------------------------------------
MODEL_PATH = "extratrees_s1_soil_moisture_points.pkl"
FEATURES_PATH = "extratrees_s1_soil_moisture_features.txt"
EXAMPLE_AOI_PATH = "examples/example_field.geojson"  # <--- NEW


# ------------------------------------------------------------
# AOI drawer map (folium) – Draw only
# ------------------------------------------------------------

def make_drawer_map_html(center_lat: float = -23.0,
                         center_lon: float = 30.0,
                         zoom: int = 7) -> str:
    """
    Returns a folium map HTML string with:
      - OSM basemap
      - Draw control (polygon only) with export to GeoJSON
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    Draw(
        export=True,
        filename="aoi.geojson",
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(m)

    return m._repr_html_()


def geocode_place(query: str):
    """
    Use OpenStreetMap Nominatim to get (lat, lon) from a place name
    for the external Gradio search box.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "giims-sm-sensor-app/1.0"}

    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"No results for '{query}'.")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon


def update_drawer_map(search_query: str) -> str:
    """
    Gradio callback to refresh the AOI drawer map.
    - If search_query is empty → default Limpopo view.
    - Else → geocode and center map on that place (zoom=13).
    """
    if not search_query or not search_query.strip():
        return make_drawer_map_html()

    try:
        lat, lon = geocode_place(search_query.strip())
        html = make_drawer_map_html(center_lat=lat, center_lon=lon, zoom=13)
        return html
    except Exception as e:
        base = make_drawer_map_html()
        msg = (
            f"<div style='color:#b91c1c;font-size:13px;margin-bottom:4px;'>"
            f"Could not find '{search_query}': {e}</div>"
        )
        return msg + base


# ------------------------------------------------------------
# Earth Engine AUTH via environment variables
# ------------------------------------------------------------

SA_EMAIL = os.environ.get(
    "EE_SERVICE_ACCOUNT",
    "zolokiala@tethys-app-1.iam.gserviceaccount.com",
)
PROJECT_ID = os.environ.get("EE_PROJECT_ID", "tethys-app-1")
EE_KEY_JSON = os.environ.get("EE_SERVICE_ACCOUNT_KEY")  # full JSON as string


def init_earth_engine():
    """
    Initialize Earth Engine using a service-account JSON stored
    in the EE_SERVICE_ACCOUNT_KEY environment variable (HF secret).
    """
    if EE_KEY_JSON is None:
        raise RuntimeError(
            "EE_SERVICE_ACCOUNT_KEY env var is not set.\n"
            "In your Hugging Face Space, go to Settings → Variables & secrets, "
            "and add EE_SERVICE_ACCOUNT_KEY with the full service-account JSON."
        )

    key_path = "/tmp/ee-service-account.json"
    if not os.path.exists(key_path):
        with open(key_path, "w") as f:
            f.write(EE_KEY_JSON)

    from ee import ServiceAccountCredentials

    credentials = ServiceAccountCredentials(SA_EMAIL, key_path)
    ee.Initialize(credentials, project=PROJECT_ID)
    print(f"✅ EE initialized: {SA_EMAIL} | project={PROJECT_ID}")


# Initialize EE BEFORE defining DEM/S1 collections
init_earth_engine()

# ---------------------------
# USER SETTINGS (constants)
# ---------------------------
MAX_DAYS_DIFF = 6
STEP_DAYS = 6
AOI_BUFFER_M = 15000
SCALE = 20

# ---------------------------
# DEM-based predictors (elev, slope)
# ---------------------------
DEM_COLL = ee.ImageCollection("COPERNICUS/DEM/GLO30")
DEM = DEM_COLL.mosaic()
DEM_ELEV = DEM.select("DEM").rename("elev")
DEM_SLOPE = ee.Terrain.slope(DEM).rename("slope")

# ---------------------------
# Sentinel-1 collection
# ---------------------------
S1_ORBIT_PASS = None  # or "ASCENDING"/"DESCENDING"


def get_s1_collection(aoi, orbit_pass=None):
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
        )
        .filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VH")
        )
    )
    if orbit_pass:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    return col


# ---------------------------
# Step-day composites
# ---------------------------

def make_s1_composites(s1_col, start_date, end_date, step_days=6):
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n = end.difference(start, "day").divide(step_days).ceil().int()

    empty = (
        ee.Image.constant([0, 0, 0])
        .rename(["VV", "VH", "angle"])
        .updateMask(ee.Image.constant(0))
    )

    def make_one(i):
        i = ee.Number(i)
        d0 = start.advance(i.multiply(step_days), "day")
        d1 = d0.advance(step_days, "day")
        win = s1_col.filterDate(d0, d1)

        comp = ee.Image(
            ee.Algorithms.If(
                win.size().gt(0),
                win.median().select(["VV", "VH", "angle"]),
                empty,
            )
        )

        mid = d0.advance(ee.Number(step_days).divide(2), "day")

        comp = comp.set(
            {
                "system:time_start": mid.millis(),
                "date": mid.format("YYYY-MM-dd"),
                "n_images": win.size(),
            }
        )
        return comp

    comps = ee.ImageCollection(
        ee.List.sequence(0, n.subtract(1)).map(make_one)
    )
    comps = comps.filter(ee.Filter.gt("n_images", 0))
    return comps


# ---------------------------
# FC -> pandas
# ---------------------------

def fc_to_pandas(fc, force_columns=None):
    d = fc.getInfo()
    rows = [f.get("properties", {}) for f in d.get("features", [])]
    df = pd.DataFrame(rows)
    print("Downloaded rows   :", len(df))
    print("Downloaded columns:", df.columns.tolist())

    if force_columns:
        for c in force_columns:
            if c not in df.columns:
                df[c] = np.nan
                print(f"⚠️ Added missing column '{c}' with NaNs.")
    return df


# ============================================================
# 1) Past-only join: composite_date <= date
# ============================================================

def attach_s1_nearest_composite_past(fc_obs, s1_comps, max_days_diff=6):
    def add_t(f):
        return f.set("t", ee.Date(f.get("date")).millis())

    fc = fc_obs.map(add_t)

    max_diff_ms = max_days_diff * 24 * 60 * 60 * 1000

    diff_filter = ee.Filter.maxDifference(
        difference=max_diff_ms,
        leftField="t",
        rightField="system:time_start",
    )

    past_filter = ee.Filter.greaterThanOrEquals(
        leftField="t", rightField="system:time_start"
    )

    filt = ee.Filter.And(diff_filter, past_filter)

    join = ee.Join.saveBest(matchKey="best_img", measureKey="time_diff")
    joined = ee.FeatureCollection(join.apply(fc, s1_comps, filt))

    matched = joined.filter(ee.Filter.notNull(["best_img"]))
    unmatched = joined.size().subtract(matched.size())
    print(
        "🔎 Join matched (server-side):",
        matched.size().getInfo(),
        "/",
        joined.size().getInfo(),
    )
    print(
        "    Unmatched (no composite within -MAX_DAYS_DIFF BEFORE date):",
        unmatched.getInfo(),
    )

    def sample_one(feat):
        img = ee.Image(feat.get("best_img"))
        full_img = img.addBands(DEM_ELEV).addBands(DEM_SLOPE)

        vals = full_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feat.geometry(),
            scale=SCALE,
            maxPixels=1e7,
        )
        return feat.set(
            {
                "VV": vals.get("VV"),
                "VH": vals.get("VH"),
                "angle": vals.get("angle"),
                "elev": vals.get("elev"),
                "slope": vals.get("slope"),
                "comp_date": img.get("date"),
                "time_diff_ms": feat.get("time_diff"),
                "n_images": img.get("n_images"),
            }
        )

    sampled = matched.map(sample_one)
    got_vv = sampled.filter(ee.Filter.notNull(["VV"])).size()
    tot = sampled.size()
    print(
        "🧪 Sampled non-null VV (server-side):",
        got_vv.getInfo(),
        "/",
        tot.getInfo(),
    )
    return sampled


# ============================================================
# 2) Build grid centroids (WGS84)
# ============================================================

def build_plot_grid_centroids(date_str, plot_geojson_path, cell_size_m):
    if not os.path.exists(plot_geojson_path):
        raise FileNotFoundError(
            f"Plot GeoJSON not found at {plot_geojson_path}."
        )

    with open(plot_geojson_path, "r") as f:
        gj = json.load(f)
    geom = ee.Geometry(gj["features"][0]["geometry"])

    bounds = geom.bounds(maxError=1)
    coords = ee.List(bounds.coordinates().get(0))

    ll = ee.List(coords.get(0))  # [minLon, minLat]
    ur = ee.List(coords.get(2))  # [maxLon, maxLat]

    min_lon = ee.Number(ll.get(0))
    min_lat = ee.Number(ll.get(1))
    max_lon = ee.Number(ur.get(0))
    max_lat = ee.Number(ur.get(1))

    cell_deg_lat = cell_size_m / 111_320.0
    cell_deg_lat_num = ee.Number(cell_deg_lat)

    lat_center = min_lat.add(max_lat).divide(2.0)
    lat_center_rad = lat_center.multiply(math.pi / 180.0)
    cos_phi = lat_center_rad.cos()
    cell_deg_lon = ee.Number(cell_size_m / 111_320.0).divide(cos_phi)

    half_lon = cell_deg_lon.divide(2.0)
    half_lat = cell_deg_lat_num.divide(2.0)

    xs = ee.List.sequence(
        min_lon.add(half_lon), max_lon.subtract(half_lon), cell_deg_lon
    )
    ys = ee.List.sequence(
        min_lat.add(half_lat), max_lat.subtract(half_lat), cell_deg_lat_num
    )

    def make_row(y):
        y = ee.Number(y)

        def make_pt(x):
            x = ee.Number(x)
            pt = ee.Geometry.Point([x, y])
            return ee.Feature(
                pt,
                {
                    "lon": x,
                    "lat": y,
                    "date": date_str,
                    "Sheet": "plot_grid",
                },
            )

        return xs.map(make_pt)

    grid_list = ys.map(make_row)
    grid_fc = ee.FeatureCollection(ee.List(grid_list).flatten())
    grid_fc = grid_fc.filterBounds(geom)
    return grid_fc, geom


# ============================================================
# 3) Predict SM on grid & compute CV
# ============================================================

def predict_sm_on_grid(date_target, plot_geojson_path, cell_size_m):
    fc_pts, geom = build_plot_grid_centroids(
        date_target, plot_geojson_path, cell_size_m
    )
    n_pts = fc_pts.size().getInfo()
    print(f"✅ Grid centroids inside plot (cell size {cell_size_m} m): {n_pts}")
    if n_pts == 0:
        raise RuntimeError(
            f"No grid centroids inside plot for cell_size_m={cell_size_m}.\n"
            "Check GeoJSON coordinates and/or reduce cell_size_m."
        )

    aoi = geom.buffer(AOI_BUFFER_M)
    s1 = get_s1_collection(aoi, S1_ORBIT_PASS)

    start_wide = (
        ee.Date(date_target)
        .advance(-MAX_DAYS_DIFF, "day")
        .format("YYYY-MM-dd")
        .getInfo()
    )
    end_wide = ee.Date(date_target).format("YYYY-MM-dd").getInfo()
    print("📅 Wide S1 date range (map):", start_wide, "to", end_wide)

    s1_period = s1.filterDate(start_wide, end_wide)
    n_s1 = s1_period.size().getInfo()
    print("🛰️ S1 images in WIDE range (map):", n_s1)
    if n_s1 == 0:
        raise RuntimeError(
            f"No S1 images in map period for this AOI (cell_size_m={cell_size_m}). "
            "Try another date or expand range."
        )

    comps = make_s1_composites(s1_period, start_wide, end_wide, STEP_DAYS)
    n_comps = comps.size().getInfo()
    print("🧱 Composites kept (non-empty, map):", n_comps)
    if n_comps == 0:
        raise RuntimeError(
            f"No non-empty composites for map inference (cell_size_m={cell_size_m}). "
            "Try a larger STEP_DAYS or date window."
        )

    fc_pts_s1 = attach_s1_nearest_composite_past(
        fc_pts, comps, MAX_DAYS_DIFF
    )
    n_pts_s1 = fc_pts_s1.size().getInfo()
    print(f"✅ Grid centroids with S1 match: {n_pts_s1} / {n_pts}")
    if n_pts_s1 == 0:
        raise RuntimeError(
            "No grid centroids could be matched to a Sentinel-1 composite in the past-only join."
        )

    df = fc_to_pandas(
        fc_pts_s1,
        force_columns=["VV", "VH", "angle", "elev", "slope", "lon", "lat"],
    )

    if len(df) == 0:
        raise RuntimeError("Joined dataframe is empty (no rows).")

    # Feature engineering (must match training)
    for col in ["VV", "VH", "angle"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["VV_VH_ratio"] = df["VV"] / df["VH"]
    df["VV_minus_VH"] = df["VV"] - df["VH"]
    df["VV_plus_VH"] = df["VV"] + df["VH"]
    df["VV_dB"] = 10.0 * np.log10(df["VV"] + 1e-6)
    df["VH_dB"] = 10.0 * np.log10(df["VH"] + 1e-6)

    if "time_diff_ms" in df.columns:
        df["time_diff_days"] = pd.to_numeric(
            df["time_diff_ms"], errors="coerce"
        ) / (1000.0 * 60.0 * 60.0 * 24.0)
    if "n_images" in df.columns:
        df["n_images"] = pd.to_numeric(df["n_images"], errors="coerce")

    for col in ["elev", "slope"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            "Model or feature file not found.\n"
            "Make sure you added to the repo:\n"
            f"  - {MODEL_PATH}\n"
            f"  - {FEATURES_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_cols = [ln.strip() for ln in f.readlines() if ln.strip()]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
            print(
                f"⚠️ Added missing feature column '{col}' with NaNs for map inference."
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        df[col] = df[col].fillna(med)

    X = df[feature_cols].values
    if X.shape[0] == 0:
        raise RuntimeError("No samples available for prediction (X has 0 rows).")

    df["sm_pred"] = model.predict(X)

    mean_sm = df["sm_pred"].mean()
    std_sm = df["sm_pred"].std(ddof=1)
    cv_pct = (std_sm / mean_sm) * 100 if mean_sm != 0 else np.nan

    print("\n=== SOIL MOISTURE UNIFORMITY (GRID CENTROIDS) ===")
    print(f"Date        : {date_target}")
    print(f"Cell size   : {cell_size_m} m")
    print(f"Mean SM     : {mean_sm:.2f}")
    print(f"Std  SM     : {std_sm:.2f}")
    print(f"CV (percent): {cv_pct:.1f}%")
    print(f"N centroids : {len(df)}")

    map_csv = f"sm_map_{date_target}_grid_{cell_size_m}m.csv"
    keep_cols = []
    for col in [
        "date",
        "lat",
        "lon",
        "elev",
        "slope",
        "VV",
        "VH",
        "angle",
        "sm_pred",
        "comp_date",
        "time_diff_days",
        "n_images",
    ]:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    out = df[keep_cols].copy()
    out.to_csv(map_csv, index=False)
    print("💾 Saved grid-centroid map CSV:", map_csv)
    print("   Rows (grid cells / centroids):", len(out))

    return cv_pct, out, geom


# ============================================================
# 4) Gradio core: run multiple grid sizes
# ============================================================

def run_sensor_optimization(date_target, geojson_file, cell_sizes_str):
    if geojson_file is None:
        msg = (
            "<b>Provide a field AOI.</b> Upload a Polygon/MultiPolygon GeoJSON (EPSG:4326), "
            "or use the AOI drawer to draw, export & upload."
        )
        raise gr.Error(msg)

    # With type="filepath", geojson_file is already a path string
    plot_geojson_path = str(geojson_file)

    try:
        cell_sizes = [int(s.strip()) for s in cell_sizes_str.split(",") if s.strip()]
    except Exception:
        raise gr.Error(
            "Could not parse grid sizes. Use a comma-separated list, e.g. '5,10,20,30'."
        )

    cvs = []
    n_sensors = []
    used_cell_sizes = []

    for cell_size in cell_sizes:
        print("\n" + "=" * 60)
        print(f"🔧 Running grid size {cell_size} m ...")
        try:
            cv_pct, df_grid, _geom = predict_sm_on_grid(
                date_target, plot_geojson_path, cell_size
            )
            cvs.append(cv_pct)
            n_sensors.append(len(df_grid))
            used_cell_sizes.append(cell_size)
        except Exception as e:
            print(f"⚠️ Skipping cell_size={cell_size} due to error: {e}")

    if len(cvs) == 0:
        raise gr.Error(
            "All grid sizes failed. Check date, GeoJSON, or model availability."
        )

    summary_df = (
        pd.DataFrame(
            {
                "cell_size_m": used_cell_sizes,
                "n_sensors": n_sensors,
                "cv_percent": cvs,
            }
        )
        .sort_values("n_sensors")
        .reset_index(drop=True)
    )

    opt_idx = int(np.nanargmin(summary_df["cv_percent"].values))
    opt_n = int(summary_df.loc[opt_idx, "n_sensors"])
    opt_cv = float(summary_df.loc[opt_idx, "cv_percent"])
    opt_cell = int(summary_df.loc[opt_idx, "cell_size_m"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary_df["n_sensors"], summary_df["cv_percent"], marker="o")
    ax.set_xlabel("Number of sensors (N centroids)")
    ax.set_ylabel("CV of soil moisture (%)")
    ax.set_title(f"CV vs Number of Sensors – {date_target}")
    ax.grid(True, alpha=0.3)

    ax.scatter([opt_n], [opt_cv], s=120, marker="*", edgecolor="black")
    ax.annotate(
        f"Optimal\nN={opt_n}\nCV={opt_cv:.1f}%",
        xy=(opt_n, opt_cv),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    ax.text(
        0.99,
        0.01,
        f"Optimal grid ≈ {opt_cell} m",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6),
    )

    plt.tight_layout()

    return fig, summary_df


# ============================================================
# 5) Centroid map (folium) + coordinates table
# ============================================================

def show_centroid_map(date_target, geojson_file, cell_size_m):
    """
    Build grid centroids for a single cell size and render them on
    a folium map, plus a coords table:
        sensor_id, Longitude (°E), Latitude (°S)
    """
    empty = pd.DataFrame(columns=["sensor_id", "Longitude (°E)", "Latitude (°S)"])

    if geojson_file is None:
        msg = (
            "<i>Please upload a field GeoJSON first, then click "
            "<b>Show centroid map</b>.</i>"
        )
        return msg, empty

    plot_geojson_path = str(geojson_file)

    try:
        cell_size_m = int(cell_size_m)
    except Exception:
        msg = "<i>Cell size must be a single integer (e.g. 10, 20, 30).</i>"
        return msg, empty

    fc_pts, geom = build_plot_grid_centroids(
        date_target, plot_geojson_path, cell_size_m
    )
    n_pts = fc_pts.size().getInfo()
    if n_pts == 0:
        msg = (
            f"<i>No grid centroids inside the plot for cell_size_m={cell_size_m} m. "
            "Try a smaller cell size or check your GeoJSON.</i>"
        )
        return msg, empty

    print(f"🗺️ Preview map: {n_pts} centroids for cell size {cell_size_m} m")

    centroid = geom.centroid().coordinates().getInfo()
    lon_c, lat_c = centroid[0], centroid[1]

    # Base map – satellite + labels style
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=16,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Esri World Imagery",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Add field polygon from uploaded GeoJSON
    try:
        with open(plot_geojson_path, "r") as f:
            gj = json.load(f)
        folium.GeoJson(
            gj,
            name="Field polygon",
            style_function=lambda x: {
                "color": "#10b981",
                "weight": 2,
                "fillOpacity": 0.05,
            },
        ).add_to(m)
    except Exception as e:
        print("⚠️ Could not add field polygon to map:", e)

    # Get centroid coordinates as pandas
    df_coords = fc_to_pandas(fc_pts, force_columns=["lon", "lat"])
    df_coords = df_coords[["lon", "lat"]].copy()
    df_coords["lon"] = pd.to_numeric(df_coords["lon"], errors="coerce").round(6)
    df_coords["lat"] = pd.to_numeric(df_coords["lat"], errors="coerce").round(6)
    df_coords.insert(0, "sensor_id", np.arange(1, len(df_coords) + 1))
    df_coords.rename(
        columns={"lon": "Longitude (°E)", "lat": "Latitude (°S)"}, inplace=True
    )

    # Add centroids as red circles
    fg = folium.FeatureGroup(name=f"Centroids ({n_pts} sensors)")
    for _, row in df_coords.iterrows():
        folium.CircleMarker(
            location=[row["Latitude (°S)"], row["Longitude (°E)"]],
            radius=4,
            color="#ef4444",
            fill=True,
            fill_opacity=0.9,
            popup=f"id={int(row['sensor_id'])}<br>"
                  f"lon={row['Longitude (°E)']}, lat={row['Latitude (°S)']}",
        ).add_to(fg)
    fg.add_to(m)

    # Simple legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(15,23,42,0.85);
        color: #f9fafb;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    ">
      <b>Soil moisture sensors</b><br>
      <span style="display:inline-block;width:10px;height:10px;
                   border-radius:50%;background:#ef4444;margin-right:4px;"></span>
      Grid centroids
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    map_html = m._repr_html_()
    return map_html, df_coords


# ============================================================
# 6) Helper: load example AOI (for demo button)
# ============================================================

def load_example_aoi():
    """
    Gradio callback that sets the GeoJSON file input to a local
    example AOI (examples/example_field.geojson).
    """
    if not os.path.exists(EXAMPLE_AOI_PATH):
        raise gr.Error(
            f"Example AOI not found at '{EXAMPLE_AOI_PATH}'. "
            "Make sure the file exists in your repo."
        )
    # For File(type="filepath"), we just return the path string
    return EXAMPLE_AOI_PATH


# ============================================================
# 7) Gradio UI – with search + Load example AOI button
# ============================================================

theme = gr.themes.Soft(
    primary_hue="teal", secondary_hue="cyan", neutral_hue="slate"
)

with gr.Blocks(
    theme=theme,
    css="""
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
}
#sm-header h1 {
    text-align: center;
}
#sm-header p {
    text-align: center;
    font-size: 0.95rem;
}
.small-note {
    font-size: 0.78rem;
    opacity: 0.8;
}
""",
) as demo:

    with gr.Column(elem_id="sm-header"):
        gr.Markdown(
            """
            # 🌱 Soil Moisture Sensor Optimization
            **Sentinel-1 + ExtraTrees – Field-scale sensor planning**

            Upload or draw a field polygon, explore different grid sizes, and find the number of
            soil moisture sensors that minimises spatial variability (CV%).
            """
        )

    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Inputs")

            date_input = gr.Textbox(
                label="Target date (YYYY-MM-DD)",
                value="2025-10-17",
                info=(
                    "Date of interest for soil moisture mapping "
                    "(must overlap Sentinel-1 coverage)."
                ),
                placeholder="e.g. 2025-10-17",
            )

            cell_sizes_input = gr.Textbox(
                label="Grid cell sizes for optimization (m, comma-separated)",
                value="5,10,20,30",
                info="Each value defines a regular grid (cell size in metres) over your field.",
                placeholder="5,10,20,30",
            )

            # IMPORTANT: type="filepath" so we can programmatically set value
            geojson_input = gr.File(
                label="Field polygon (GeoJSON; EPSG:4326, Polygon/MultiPolygon)",
                file_types=[".geojson"],
                file_count="single",
                type="filepath",
            )

            # New "Load example AOI" button
            example_button = gr.Button(
                "📂 Load example AOI",
                variant="secondary",
            )

            with gr.Accordion(
                "Draw / Search AOI (folium Draw) — export & upload here", open=False
            ):
                search_box = gr.Textbox(
                    label="Search place (optional)",
                    placeholder="e.g. Groblersdal, South Africa",
                    info=(
                        "Type a place name and click 'Search & update AOI map' "
                        "to centre the AOI drawer."
                    ),
                )
                search_button = gr.Button("🔍 Search & update AOI map")

                drawer_map_html = gr.HTML(
                    value=make_drawer_map_html(), label="AOI drawer map"
                )

                gr.Markdown(
                    """
                    <div class="small-note">
                    1. Use the search box above or just pan/zoom on the map.<br>
                    2. Draw a polygon with the draw tools (top-left).<br>
                    3. Use the <b>Export</b> button in the draw toolbar to download <code>aoi.geojson</code>.<br>
                    4. Upload that file in the <b>Field polygon</b> input above — or click <b>Load example AOI</b>.
                    </div>
                    """,
                    elem_classes=["small-note"],
                )

            run_button = gr.Button("▶ Run sensor optimization", variant="primary")

            gr.Markdown(
                """
                <div class="small-note">
                💡 <b>Quick start:</b> Click <b>Load example AOI</b> → run optimization.  
                Or: Search/draw your own field → export GeoJSON → upload it → run optimization.
                </div>
                """,
                elem_classes=["small-note"],
            )

        # Right column
        with gr.Column(scale=1.2):
            with gr.Tabs():
                with gr.Tab("Optimization"):
                    gr.Markdown("### 📊 CV vs Number of Sensors")

                    plot_output = gr.Plot(label="CV vs Number of Sensors")

                    table_output = gr.Dataframe(
                        label="Summary by grid size",
                        headers=["cell_size_m", "n_sensors", "cv_percent"],
                        interactive=False,
                    )

                    gr.Markdown(
                        """
                        <div class="small-note">
                        The optimal configuration is marked with a star ⭐ on the graph, and corresponds to the
                        lowest coefficient of variation (CV%) in predicted soil moisture.
                        </div>
                        """,
                        elem_classes=["small-note"],
                    )

                with gr.Tab("Sensor layout preview"):
                    gr.Markdown("### 🗺️ Centroids and coordinates")

                    map_cell_size_input = gr.Dropdown(
                        label="Grid cell size for map (m)",
                        choices=[5, 10, 20, 30],
                        value=10,
                        interactive=True,
                        info="Choose one grid size to preview centroid locations.",
                    )

                    map_button = gr.Button(
                        "Show centroid map", variant="secondary"
                    )

                    map_html_output = gr.HTML(
                        label="Field polygon and centroids"
                    )

                    centroid_table_output = gr.Dataframe(
                        label=(
                            "Centroid coordinates "
                            "(sensor_id, Longitude (°E), Latitude (°S))"
                        ),
                        interactive=False,
                    )

                    gr.Markdown(
                        """
                        <div class="small-note">
                        Sensor positions shown here are centroids only (no soil-moisture map), with coordinates
                        you can copy into field sheets or Kobo forms.
                        </div>
                        """,
                        elem_classes=["small-note"],
                    )

    gr.Markdown(
        """
        ---
        <div class="small-note">
        Prototype developed around GIIMS soil-moisture workflow. Exported CSVs (per grid size)
        can be used for further analysis or for designing field experiments.
        </div>
        """,
        elem_classes=["small-note"],
    )

    # Wiring
    run_button.click(
        fn=run_sensor_optimization,
        inputs=[date_input, geojson_input, cell_sizes_input],
        outputs=[plot_output, table_output],
    )

    map_button.click(
        fn=show_centroid_map,
        inputs=[date_input, geojson_input, map_cell_size_input],
        outputs=[map_html_output, centroid_table_output],
    )

    search_button.click(
        fn=update_drawer_map,
        inputs=[search_box],
        outputs=[drawer_map_html],
    )

    # NEW: Load example AOI into File component
    example_button.click(
        fn=load_example_aoi,
        inputs=None,
        outputs=[geojson_input],
    )

# For HF Spaces and local testing
if __name__ == "__main__":
    demo.launch()
