import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static

# --- CONFIG ---
CSV_PATH = 'indonesia_data.csv'
SHAPEFILE_DIR = r"C:/Users/chaim/OneDrive/Bureau/mapping2"

st.title("Indonesia COVID-19 New Cases Choropleth Map")

# Helper function to safely display GeoDataFrame (without geometry issues)
def safe_display(df):
    df_copy = df.copy()
    if 'geometry' in df_copy.columns:
        df_copy['geometry'] = df_copy['geometry'].astype(str)
    return df_copy

# 1. Load data
@st.cache_data
def load_data(csv_path, shapefile_path):
    try:
        # Load CSV
        df = pd.read_csv(csv_path, sep=';')
        df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
        df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
        
        # Create pivot table
        pivot = df.pivot(index='Date', columns='Location', values='New Cases').reset_index()
        
        # Load shapefile
        shp = gpd.read_file(shapefile_path)
        
        # Display available columns in shapefile for debugging
        st.sidebar.write("Available columns in shapefile:", shp.columns.tolist())
        
        # Match columns that exist in both datasets
        # Use NAME_1 since it exists in your shapefile
        valid_names = set(pivot.columns) & set(df['Location'].unique())
        cols = [c for c in pivot.columns if c == 'Date' or c in valid_names]
        data_clean = pivot[cols]
        
        # Process for map
        data_t = data_clean.set_index('Date').T
        data_t.index.name = 'Province'
        
        # Create a mapping dictionary between province names and shapefile NAME_1
        # This requires manual verification to ensure correct matching
        
        # Join with shapefile directly using NAME_1
        merged_gdf = None
        for date in data_clean['Date']:
            date_data = data_clean.set_index('Date').loc[date].reset_index()
            date_data.columns = ['Province', 'New_Cases']
            
            # Merge with shapefile for this date
            temp_gdf = shp.merge(date_data, left_on='NAME_1', right_on='Province', how='left')
            temp_gdf[date] = temp_gdf['New_Cases']
            
            if merged_gdf is None:
                merged_gdf = temp_gdf[['NAME_1', 'geometry', date]]
            else:
                merged_gdf[date] = temp_gdf[date]
        
        return data_clean['Date'].tolist(), merged_gdf, data_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)
        return [], None, None

# Load data
dates, gdf, data_clean = load_data(CSV_PATH, f"{SHAPEFILE_DIR}/IDN_Indonesia_1.shp")

if not dates or gdf is None:
    st.error("Failed to load data. Please check your file paths and data format.")
    st.stop()

# 2. Display date selector
dates = sorted(dates)
selected_date = st.select_slider("Select date:", options=dates, value=dates[-1])

# 3. Create map
def make_map(gdf, date):
    # Create map
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="cartodbpositron")
    
    # Add choropleth - use NAME_1 which is confirmed to exist
    choropleth = folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        name="choropleth",
        data=gdf,
        columns=['NAME_1', date],
        key_on='feature.properties.NAME_1',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"New COVID-19 Cases ({date})",
        highlight=True
    ).add_to(m)
    
    # Add tooltips with province name and case count
    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}
    
    folium.GeoJson(
        gdf,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NAME_1', date],
            aliases=['Province:', 'New cases:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
        )
    ).add_to(m)
    
    # Add case numbers directly on the map for each province
    for idx, row in gdf.iterrows():
        # Get the province name and case count
        province_name = row['NAME_1']
        cases = row[date]
        
        # Only add labels for provinces with data
        if pd.notna(cases) and cases > 0:
            # Format the case count nicely
            cases_formatted = f"{int(cases):,}"
            
            # Add a marker at the centroid of each province with the case count
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size: 10px;
                        font-weight: bold;
                        background-color: white;
                        border: 1px solid #666;
                        border-radius: 3px;
                        padding: 1px 3px;
                        white-space: nowrap;
                        text-align: center;
                        box-shadow: 0 0 3px rgba(0,0,0,0.3);">
                        {cases_formatted}
                    </div>
                    """
                )
            ).add_to(m)
    
    return m

# Create and display map
st.subheader(f"Choropleth for {selected_date}")
try:
    if gdf is not None and selected_date in gdf.columns:
        m = make_map(gdf, selected_date)
        folium_static(m, width=700, height=500)
    else:
        st.error(f"Selected date {selected_date} not available in the data")
        st.write("Available columns:", gdf.columns.tolist() if gdf is not None else "None")
except Exception as e:
    st.error(f"Error creating map: {e}")
    st.exception(e)
    if gdf is not None:
        st.write("Available columns:", gdf.columns.tolist())
        st.write("Sample data:", safe_display(gdf.head()))

# Optional: Display data table
if st.checkbox("Show data table"):
    if data_clean is not None:
        st.dataframe(data_clean.set_index('Date').loc[selected_date].sort_values(ascending=False))
    else:
        st.error("No data available")

# Add a debugging section
if st.checkbox("Show debug information"):
    st.subheader("Debug Information")
    if gdf is not None:
        st.write("GeoDataFrame columns:", gdf.columns.tolist())
        st.write("Selected date column exists:", selected_date in gdf.columns)
        st.write("First few rows of GeoDataFrame:")
        st.write(safe_display(gdf.head()))
    else:
        st.error("GeoDataFrame is None")
