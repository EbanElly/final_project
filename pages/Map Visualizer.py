import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import warnings

st.set_page_config(page_title="Tanzania Disease Tracker", layout="wide")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Check Authentication ---

# Check if user is logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.markdown('<div class="stHeader">Access Denied</div>', unsafe_allow_html=True)
    st.markdown('<div class="stText">Please <a href="/">log in</a> or sign up to access this page.</div>', unsafe_allow_html=True)
    st.stop()

# --- Sidebar ---
st.sidebar.markdown(f'<div class="stText">Welcome, {st.session_state["username"]}!</div>', unsafe_allow_html=True)
if st.sidebar.button("Logout"):
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    try:
        st.switch_page("Home.py")  # Redirect to the home page (adjust if your home page is named differently, e.g., "Home.py")
    except AttributeError:
        st.rerun()  # Fallback if switch_page is not available (older Streamlit versions)

# --- Page Config ---
st.title("🦠 Waterborne Diseases Cases Map View")

# --- Manual Region Mapping ---
REGION_MAPPING = {
    "dar-es-salaam": "dar es salaam",  # GeoJSON to CSV
    "arusha": "arusha",
    "dodoma": "dodoma",
    "geita": "geita",
    "iringa": "iringa",
    "kagera": "kagera",
    "katavi": "katavi",
    "kigoma": "kigoma",
    "kilimanjaro": "kilimanjaro",
    "lindi": "lindi",
    "manyara": "manyara",
    "mara": "mara",
    "mbeya": "mbeya",
    "morogoro": "morogoro",
    "mtwara": "mtwara",
    "mwanza": "mwanza",
    "njombe": "njombe",
    "pemba kaskazini": "pemba kaskazini",
    "pemba kusini": "pemba kusini",
    "pwani": "pwani",
    "rukwa": "rukwa",
    "ruvuma": "ruvuma",
    "shinyanga": "shinyanga",
    "simiyu": "simiyu",
    "singida": "singida",
    "tabora": "tabora",
    "tanga": "tanga",
    "zanzibar kaskazini": "zanzibar kaskazini",
    "zanzibar kusini and kati": "zanzibar kusini and kati",
    "zanzibar magharibi": "zanzibar magharibi"
}

# --- Data Loading ---
@st.cache_resource
def load_data(csv_path="data/Tanzania_Health_Data_Updated.csv", geojson_path="data/geoBoundaries-TZA-ADM1.geojson"):
    try:
        # Load disease data
        df = pd.read_csv(csv_path)
        
        # Validate CSV data
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Cholera_Cases"] = pd.to_numeric(df["Cholera_Cases"], errors="coerce").astype("Int64")
        df["Typhoid_Cases"] = pd.to_numeric(df["Typhoid_Cases"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Year", "Cholera_Cases", "Typhoid_Cases"])
        
        # Clean region names
        df["region_key"] = df["Region"].str.lower().str.strip()
        
        # Load GeoJSON
        gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
        gdf["region_key"] = gdf["shapeName"].str.lower().str.strip()
        
        # Check for null geometries
        if gdf.geometry.isna().any():
            st.error("GeoJSON contains null geometries, which prevents mapping. Please use a GeoJSON with valid Polygon/MultiPolygon geometries.")
            return None, None
        
        # Check for missing regions
        csv_regions = set(df["region_key"])
        geojson_regions = set(gdf["region_key"])
        missing_regions = csv_regions - geojson_regions
        
        # Apply region mapping
        gdf["region_key_mapped"] = gdf["region_key"].map(REGION_MAPPING)
        
        return df, gdf
    except FileNotFoundError as e:
        st.error(f"Error: Could not find file - {e}")
        return None, None
    except KeyError as e:
        st.error(f"Error: Missing expected field in GeoJSON (e.g., shapeName) - {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load data
df, regions_gdf = load_data()

# Check if data loaded successfully
if df is None or regions_gdf is None:
    st.error("Please ensure 'Tanzania_Health_Data_Updated.csv' and 'geoBoundaries-TZA-ADM1.geojson' are in the same directory.")
    st.stop()

# --- UI Controls ---
col1, col2 = st.columns(2)
with col1:
    disease = st.selectbox("Select Disease", options=["Cholera_Cases", "Typhoid_Cases"], index=None, format_func=lambda x: x.replace("_", " "))
with col2:
    year = st.selectbox("Select Year", options=[2020, 2021, 2022, 2023, 2024], index=None)

# --- Data Processing ---
def prepare_map_data(df, gdf, year, disease):
    try:
        # Filter by year
        filtered = df[df["Year"] == year].copy()
        
        if filtered.empty:
            st.warning(f"Fill all selection boxes")
            return None
        
        # Merge with GeoJSON data
        merged = gdf.merge(
            filtered,
            how="left",
            left_on="region_key_mapped",
            right_on="region_key"
        )
        
        # Fill missing values
        merged[disease] = merged[disease].fillna(0).astype(int)
        
        return merged
    except Exception as e:
        st.error(f"Error preparing map data: {e}")
        return None

# Prepare visualization data
viz_data = prepare_map_data(df, regions_gdf, year, disease)

# --- Visualization ---
if viz_data is not None and not viz_data.empty and disease and year:
    # Calculate max cases for color scaling
    max_cases = viz_data[disease].max()
    
    # Create color scale
    colormap = cm.LinearColormap(
        colors=["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
        vmin=0,
        vmax=max_cases if max_cases > 0 else 1,
        caption=f"{disease.replace('_', ' ')} Cases (0 to {max_cases})"
    )
    
    # Define color ranges with descriptive labels
    def get_color_ranges(max_cases):
        ranges = []
        if max_cases <= 0:
            return [("No Cases", 0, 0, "#ffffcc")]
        step = max_cases / 4
        colors = ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]
        for i in range(5):
            lower = int(i * step)
            upper = int((i + 1) * step) if i < 4 else int(max_cases)
            color = colors[i]
            label = f"[{lower} - {upper} cases]"
            ranges.append((label, lower, upper, color))
        return ranges

    color_ranges = get_color_ranges(max_cases)
    
    # Create map centered on Tanzania
    m = folium.Map(location=[-6.3690, 34.8888], zoom_start=5.5, tiles="cartodbpositron")
    
    # Add choropleth layer
    folium.GeoJson(
        viz_data,
        name="Disease Outbreaks",
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"][disease] if feature["properties"][disease] else 0),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["shapeName", disease],
            aliases=["Region:", f"{disease.replace('_', ' ')}:"],
            localize=True
        )
    ).add_to(m)
    
    # Add colorbar legend
    colormap.add_to(m)
    
    # Custom legend for color ranges with improved visibility
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 20px; z-index: 1000; background-color: white; padding: 15px; border: 2px solid #333; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); font-size: 16px; color: #333; max-width: 200px;">
        <strong>Legend</strong><br>
    """
    for label, lower, upper, color in color_ranges:
        legend_html += f'<div style="margin: 5px 0; display: flex; align-items: center;"><span style="display: inline-block; width: 20px; height: 20px; background-color: {color}; margin-right: 10px;"></span>{label}</div>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Fit map to bounds
    m.fit_bounds(m.get_bounds())
    
    # Display map
    st.subheader(f"{disease.replace('_', ' ')} Outbreaks in {year}")
    st_folium(m, width=1200, height=700)
    
    # Show interesting fact
    top_region = viz_data.loc[viz_data[disease].idxmax(), "shapeName"]
    top_cases = viz_data[disease].max()
    st.markdown(f"**Interesting Fact**: In {year}, {top_region} had the highest number of {disease.replace('_', ' ').lower()} with {top_cases} cases, highlighting a potential hotspot for targeted interventions.")
    
    # Attribution
    st.markdown("**Data Sources**: Disease data from Tanzania_Health_Data_Updated.csv; Region boundaries from [geoBoundaries](https://www.geoboundaries.org) via [HDX](https://data.humdata.org).")
else:
    if not disease or not year:
        st.warning("Please select both a disease and a year to view the map.")
    else:
        st.warning("No data available for the selected filters. Please check your GeoJSON file or region names.")