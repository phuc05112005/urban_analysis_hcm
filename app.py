# app.py

import os
import glob
import unicodedata
import math
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import geopandas as gpd
import ee
from geemap import foliumap
import json

# ================== CẤU HÌNH ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_ZIP = os.path.join(BASE_DIR, "vn_shp.zip")  # file zip shapefile
SHP_DIR = os.path.join(BASE_DIR, "vn_shp")     # thư mục giải nén

# Giải nén shapefile nếu chưa có
if not os.path.exists(SHP_DIR):
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(SHP_ZIP, "r") as zip_ref:
        zip_ref.extractall(SHP_DIR)

# ================== GOOGLE EARTH ENGINE ==================

gee_key_json = st.secrets["GEE_KEY"]
with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
    f.write(gee_key_json)
    SERVICE_ACCOUNT_FILE = f.name

@st.cache_resource(show_spinner=False)
def init_ee(key_path):
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/earthengine.readonly"]
    )
    try:
        ee.Initialize(credentials)
    except Exception:
        ee.Initialize(credentials, use_cloud_api=True)
    return True

init_ee(SERVICE_ACCOUNT_FILE)

# ================== PRESET ==================

PRESET_MEM = {
    "ui_percentile": 60,
    "ndvi_max": 0.60,
    "ndbi_min": -0.1,
    "mndwi_max": 0.05,
    "min_area_m2": 800
}

# ================== SENTINEL-2 ==================

def s2_mask(img):
    scl = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(mask)

@st.cache_data(show_spinner=False)
def get_s2_image(roi_geojson, year, cloud_thresh=40):
    roi = ee.Geometry(roi_geojson)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(roi)
           .filterDate(f"{year}-01-01", f"{year}-12-31")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
           .map(s2_mask))
    img = col.median().clip(roi)
    return img.select(
        ["B2","B3","B4","B8","B11","B12"],
        ["BLUE","GREEN","RED","NIR","SWIR1","SWIR2"]
    )

# ================== CHỈ SỐ ==================

def add_indices(img):
    ndvi = img.normalizedDifference(["NIR","RED"]).rename("NDVI")
    ndbi = img.normalizedDifference(["SWIR1","NIR"]).rename("NDBI")
    mndwi = img.normalizedDifference(["GREEN","SWIR1"]).rename("MNDWI")
    ui = ndbi.subtract(ndvi).rename("UI")
    return img.addBands([ndvi, ndbi, mndwi, ui])

def get_ui_threshold_fast(ui_img, water_mask, roi,
                          percentile=70, sample_scale=30, n=5000, seed=1):
    ui_valid = ui_img.updateMask(water_mask.lt(0))
    pts = ui_valid.sample(region=roi, scale=sample_scale, numPixels=2000, seed=seed, geometries=False)
    stats = pts.reduceColumns(reducer=ee.Reducer.percentile([int(percentile)]), selectors=["UI"])
    return ee.Number(ee.Dictionary(stats).values().get(0))

def get_urban(img, roi, params=PRESET_MEM, ui_threshold_fixed=None):
    ui = img.select("UI")
    ndvi = img.select("NDVI")
    ndbi = img.select("NDBI")
    mndwi = img.select("MNDWI")

    if ui_threshold_fixed is None:
        ui_thr = get_ui_threshold_fast(ui, mndwi, roi,
                                       percentile=params["ui_percentile"],
                                       sample_scale=30, n=5000)
    else:
        ui_thr = ee.Number(ui_threshold_fixed)

    urban_raw = (ui.gte(ui_thr)
                 .And(ndvi.lt(params["ndvi_max"]))
                 .And(ndbi.gt(params["ndbi_min"]))
                 .And(mndwi.lt(params["mndwi_max"]))).rename("URBAN_RAW")

    native_px = 10
    proc_scale = 60
    min_px = math.ceil(params["min_area_m2"] / (proc_scale ** 2))

    urban_resampled = urban_raw.reproject(crs=img.projection(), scale=proc_scale)
    connected = urban_resampled.connectedPixelCount(256, True)
    urban_filtered = urban_resampled.updateMask(connected.gte(min_px)).rename("URBAN")
    urban_final = urban_filtered.reproject(crs=img.projection(), scale=native_px)

    return urban_final, ui_thr

# ================== FIX TIẾNG VIỆT ==================

VIET_CHARS = "ĂÂÊÔƠƯăâêôơưĐđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"

def _fix_mojibake_utf8(s):
    if not isinstance(s, str):
        return s
    try:
        fixed = s.encode("latin1").decode("utf-8")
        def score(t): return sum(ch in VIET_CHARS for ch in t)
        return fixed if score(fixed) >= score(s) else s
    except Exception:
        return s

@st.cache_data(show_spinner=False)
def load_shapefile():
    shp_list = glob.glob(os.path.join(SHP_DIR, "*.shp"))
    if not shp_list:
        raise FileNotFoundError("Không tìm thấy file .shp trong vn_shp/")
    shp_path = Path(shp_list[0])
    cpg_path = shp_path.with_suffix(".cpg")
    if not cpg_path.exists():
        with open(cpg_path, "w", encoding="ascii") as f:
            f.write("UTF-8")
    gdf = gpd.read_file(str(shp_path))
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(4326)
    cols_lower = [c.lower() for c in gdf.columns]
    preferred = ["name","ten_tinh","tinh","ten","province"]
    name_field = None
    for k in preferred:
        if k in cols_lower:
            name_field = gdf.columns[cols_lower.index(k)]
            break
    if not name_field:
        text_cols = [c for c in gdf.columns if gdf[c].dtype == object]
        if not text_cols:
            raise ValueError("Không tìm thấy trường tên trong shapefile")
        name_field = text_cols[0]
    sample = str(gdf[name_field].iloc[0])
    if any(x in sample for x in ["Ã","Â","ê°","àº","»"]):
        gdf[name_field] = gdf[name_field].apply(_fix_mojibake_utf8)
    gdf[name_field] = gdf[name_field].apply(lambda s: unicodedata.normalize("NFC", s) if isinstance(s, str) else s)
    return gdf, name_field

def clean_geometry(geom):
    ee_geom = ee.Geometry(geom) if not isinstance(geom, ee.Geometry) else geom
    return ee_geom.buffer(50).buffer(-50)

# ================== GIAO DIỆN ==================

st.set_page_config(page_title="Đô thị hoá TP.HCM", layout="wide")
st.title("Công cụ đánh giá đô thị hoá TP.HCM")

try:
    gdf, name_field = load_shapefile()
except Exception as e:
    st.error(f"Lỗi shapefile: {e}")
    st.stop()

province = "VNSG"  # TP.HCM
sel = gdf[gdf[name_field] == province]
if sel.empty:
    st.error(f"Không tìm thấy {province} trong shapefile")
    st.stop()

geom = sel.geometry.unary_union.__geo_interface__
roi = ee.Geometry(geom)
roi_simple = clean_geometry(roi).simplify(100)
roi_geojson = roi_simple.getInfo()

# ================== CHỌN NĂM ==================

years = list(range(2019, 2026))
col1, col2 = st.columns(2)
with col1:
    year1 = st.selectbox("🕒 Chọn năm 1:", years, index=2)
with col2:
    year2 = st.selectbox("🕒 Chọn năm 2:", years, index=5)

# ================== PHÂN TÍCH ==================

if st.button("🚀 Bắt đầu phân tích", use_container_width=True):
    results_all = {}
    for year in [year1, year2]:
        base = get_s2_image(roi_geojson, year)
        img = add_indices(base)
        urban, ui_thr_val = get_urban(img, roi_simple, PRESET_MEM)
        ui_thr_val = float(ui_thr_val.getInfo())

        area_img = ee.Image.pixelArea().divide(1e6)
        urban_mask = urban.gt(0).selfMask()

        urban_area_dict = area_img.updateMask(urban_mask).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_simple,
            scale=120,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4
        )

        total_area_dict = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_simple,
            scale=120,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4
        )

        mean_stats_dict = img.select(["NDVI","NDBI","MNDWI","UI"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi_simple,
            scale=120,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4
        )

        urban_area_val = urban_area_dict.get("area").getInfo() if urban_area_dict.get("area") else 0.0
        total_area_val = total_area_dict.get("area").getInfo() if total_area_dict.get("area") else 0.0
        mean_stats_val = {k: float(v) if v is not None else 0.0 for k, v in mean_stats_dict.getInfo().items()}

        results_all[year] = {
            "urban_area": float(urban_area_val),
            "total_area": float(total_area_val),
            "mean_stats": mean_stats_val,
            "img": img,
            "urban": urban,
        }

    # ================== HIỂN THỊ ==================
    st.subheader("📊 Kết quả đô thị hoá TP.HCM")
    cols_res = st.columns(2)
    mean_stats_year1 = results_all[year1]["mean_stats"]
    urban_area_year1 = results_all[year1]["urban_area"]
    total_area_year1 = results_all[year1]["total_area"]
    pct_year1 = (urban_area_year1 / total_area_year1 * 100) if total_area_year1 > 0 else 0.0

    for i, year in enumerate([year1, year2]):
        res = results_all[year]
        urban_area_km2 = res["urban_area"]
        total_area_km2 = res["total_area"]
        pct = (urban_area_km2 / total_area_km2 * 100) if total_area_km2 > 0 else 0.0
        mean_ndvi = res["mean_stats"].get("NDVI")
        mean_ndbi = res["mean_stats"].get("NDBI")
        mean_mndwi = res["mean_stats"].get("MNDWI")
        mean_ui = res["mean_stats"].get("UI")

        with cols_res[i]:
            st.markdown(f"### Năm {year}")
            st.write(f"🌆 Diện tích đô thị: **{urban_area_km2:,.2f} km²**")
            st.write(f"🌍 Tổng diện tích TP.HCM: **{total_area_km2:,.2f} km²**")
            if year == year2:
                def delta_line(val1, val2, label="", unit=""):
                    diff = val2 - val1
                    color = "green" if diff >= 0 else "red"
                    sign = "+" if diff >= 0 else ""
                    return f'<span style="color:{color}">{label}({sign}{diff:.3f}{unit})</span>'
                st.markdown(f"📈 Tỷ lệ đô thị hoá: **{pct:.2f}%** {delta_line(pct_year1, pct, '→ ', '%')}", unsafe_allow_html=True)
                st.markdown(f"🟩 NDVI: **{mean_ndvi:.3f}** {delta_line(mean_stats_year1['NDVI'], mean_ndvi, '→ ')}", unsafe_allow_html=True)
                st.markdown(f"🟫 NDBI: **{mean_ndbi:.3f}** {delta_line(mean_stats_year1['NDBI'], mean_ndbi, '→ ')}", unsafe_allow_html=True)
                st.markdown(f"💧 MNDWI: **{mean_mndwi:.3f}** {delta_line(mean_stats_year1['MNDWI'], mean_mndwi, '→ ')}", unsafe_allow_html=True)
                st.markdown(f"⚙️ UI: **{mean_ui:.3f}** {delta_line(mean_stats_year1['UI'], mean_ui, '→ ')}", unsafe_allow_html=True)
            else:
                st.write(f"📈 Tỷ lệ đô thị hoá: **{pct:.2f}%**")
                st.write(f"🟩 NDVI: **{mean_ndvi:.3f}**")
                st.write(f"🟫 NDBI: **{mean_ndbi:.3f}**")
                st.write(f"💧 MNDWI: **{mean_mndwi:.3f}**")
                st.write(f"⚙️ UI: **{mean_ui:.3f}**")

            m = foliumap.Map(height=500)
            m.add_basemap("HYBRID")
            m.centerObject(roi_simple, 8)
            m.addLayer(res["urban"].updateMask(res["urban"]), {"palette": ["#ff0000"]}, "Khu đô thị")
            m.addLayer(res["img"].select("NDVI"), {"min": 0, "max": 1, "palette": ["#FFFFFF", "#befac0", "#54bf59", "#3cbe42", "#0f9d18"]}, "NDVI")
            m.addLayer(res["img"].select("NDBI"), {"min": -0.3, "max": 0.4, "palette": ["#FFFFFF", "#DFB68C", "#834729", "#793C14", "#683008"]}, "NDBI")
            m.addLayer(res["img"].select("MNDWI"), {"min": -0.5, "max": 0.5, "palette": ["#FFFFFF", "#a7d9f4", "#4596d0", "#286eb5", "#08306b"]}, "MNDWI")
            m.addLayer(res["img"].select("UI"), {"min": -0.5, "max": 0.5, "palette": ["#FFFFFF", "#ec5b5b", "#c63030", "#851e1a", "#680004"]}, "UI")
            roi_fc = ee.FeatureCollection(roi_simple).style(color="white", fillColor="00000000", width=2)
            m.addLayer(roi_fc, {}, "Ranh giới")
            m.addLayerControl()
            m.to_streamlit()
