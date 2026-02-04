# Soil Moisture Sensor Optimization (Sentinel-1 + ExtraTrees)

---
title: soil-moisture-sensor-optimizer
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---



Interactive app to plan soil moisture sensor layouts for a field:

- Upload a **field polygon** (GeoJSON) or draw an AOI on a map.
- For a given date, the app:
  - Builds regular grids at multiple cell sizes (e.g. 5, 10, 20, 30 m).
  - Samples Sentinel-1 + DEM at the grid centroids.
  - Uses a trained **ExtraTrees** model to predict soil moisture.
  - Computes the coefficient of variation (CV%) across centroids.
  - Finds the **optimal number of sensors** (min CV).

## Running on Hugging Face Spaces

1. **Create a new Space**

   - Space type: **Gradio**
   - SDK: `Gradio`
   - Python version: 3.10 or 3.11 is fine.

2. **Add the repo files**

   - Upload:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `.gitignore` (optional)
   - Add your trained model & feature list to the root:
     - `extratrees_s1_soil_moisture_points.pkl`
     - `extratrees_s1_soil_moisture_features.txt`

3. **Configure Earth Engine credentials**

   In the Space, go to **Settings → Variables and secrets** and add:

   - `EE_SERVICE_ACCOUNT` – your service account email  
     e.g. `zolokiala@tethys-app-1.iam.gserviceaccount.com`
   - `EE_PROJECT_ID` – your Earth Engine project ID  
     e.g. `tethys-app-1`
   - `EE_SERVICE_ACCOUNT_KEY` – **entire service-account JSON** as a string  
     (open the `.json` key file, copy all its text, paste as the secret value).

   The app writes this JSON to `/tmp/ee-service-account.json` and uses it to initialize Earth Engine.

4. **Build & run**

   Once you commit / push, the Space will install dependencies from `requirements.txt`
   and automatically serve the `demo` Gradio app defined in `app.py`.

## Local development

```bash
git clone https://github.com/<your-user>/soil-moisture-sensor-optimizer.git
cd soil-moisture-sensor-optimizer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# set env vars locally (e.g. via export or a .env file)
export EE_SERVICE_ACCOUNT="your-sa-email@project.iam.gserviceaccount.com"
export EE_PROJECT_ID="your-ee-project-id"
export EE_SERVICE_ACCOUNT_KEY="$(cat /path/to/service-account.json)"

python app.py
