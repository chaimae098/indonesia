import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from datetime import datetime
import plotly.express as px
import os

# --- CONFIG ---
CSV_PATH = 'data/indonesia_data.csv'
SHAPEFILE_DIR = 'data'

# Set page configuration
st.set_page_config(
    page_title="Indonesia COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        padding-bottom: 20px;
    }
    h2, h3 {
        color: #2563EB;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stSlider > div > div > div {
        background-color: #EF4444;
    }
    /* Improves the date slider appearance */
    .stSlider [data-baseweb="slider"] {
        height: 6px;
    }
    /* Styles date text */
    div[data-testid="stExpander"] p {
        font-size: 16px;
    }
    /* Card styling for map */
    .map-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title with better styling
st.title("Indonesia COVID-19 New Cases Choropleth Map")

# Helper function to safely display GeoDataFrame
def safe_display(df):
    df_copy = df.copy()
    if 'geometry' in df_copy.columns:
        df_copy['geometry'] = df_copy['geometry'].astype(str)
    return df_copy

# 1. Improved data loading strategy with better caching
@st.cache_data(ttl=3600)
def load_base_data(csv_path):
    """Load and preprocess just the raw data"""
    try:
        # Load CSV with direct date parsing
        df = pd.read_csv(csv_path, sep=';', parse_dates=['Date'])
        df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
        df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
        
        # Get unique dates in chronological order
        dates = sorted(df['Date'].unique())
        date_strings = [d.strftime('%m/%d/%Y') for d in dates]
        
        return df, dates, date_strings
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return None, [], []

@st.cache_data(ttl=3600)
def load_shapefile(shapefile_path):
    """Load just the shapefile separately"""
    try:
        # Check if file exists first
        if not os.path.exists(shapefile_path):
            st.error(f"Shapefile not found at: {shapefile_path}")
            return None
        return gpd.read_file(shapefile_path)
    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        st.exception(e)
        return None

# Show loading message while data is being prepared
with st.spinner("Loading COVID-19 data..."):
    # Load base data
    raw_df, date_objects, dates = load_base_data(CSV_PATH)
    
    # Display shapefile path for debugging
    shapefile_path = f"{SHAPEFILE_DIR}/IDN_Indonesia_1.shp"
    if not os.path.exists(shapefile_path):
        st.error(f"Shapefile not found at: {shapefile_path}")
        st.write("Current directory:", os.getcwd())
        st.write("Files in data directory:", os.listdir(SHAPEFILE_DIR) if os.path.exists(SHAPEFILE_DIR) else "Data directory not found")
    
    # Try alternative common shapefile names if exact path is not found
    shapefile = None
    if not os.path.exists(shapefile_path):
        alternatives = [
            f"{SHAPEFILE_DIR}/IDN_adm1.shp",  # GADM format
            f"{SHAPEFILE_DIR}/indonesia.shp",
            f"{SHAPEFILE_DIR}/idn_adm1.shp",
            f"{SHAPEFILE_DIR}/IDN.shp"
        ]
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                st.info(f"Using alternative shapefile: {alt_path}")
                shapefile = load_shapefile(alt_path)
                break
    else:
        # Load shapefile separately
        shapefile = load_shapefile(shapefile_path)

if raw_df is None or shapefile is None:
    st.error("Failed to load data. Please check your file paths and data format.")
    st.stop()

# Process data for a specific date (only when needed)
@st.cache_data(ttl=3600)
def get_province_data(raw_df, date_obj):
    """Get province data for a specific date"""
    try:
        # Filter data for this date
        date_data = raw_df[raw_df['Date'] == date_obj].copy()
        
        # Create province-level data
        province_data = date_data[['Location', 'New Cases']].rename(
            columns={'Location': 'Province', 'New Cases': 'New_Cases'}
        )
        
        return date_obj.strftime('%m/%d/%Y'), province_data
    except Exception as e:
        st.error(f"Error processing data for date {date_obj}: {e}")
        return None, None

def get_date_data(raw_df, shapefile, date_obj):
    """Get merged geodataframe for a specific date"""
    try:
        # Get province data (cached)
        date_str, province_data = get_province_data(raw_df, date_obj)
        
        if date_str is None or province_data is None:
            return None, None, None
            
        # Merge with shapefile just for this date - this part isn't cached
        # because geopandas DataFrames aren't hashable for st.cache_data
        merged_gdf = shapefile.merge(province_data, left_on='NAME_1', right_on='Province', how='left')
        merged_gdf[date_str] = merged_gdf['New_Cases']
        
        return merged_gdf, date_str, province_data
    except Exception as e:
        st.error(f"Error processing data for date {date_obj}: {e}")
        return None, None, None

# 2. Display date selector with chronological ordering
min_date = dates[0]
max_date = dates[-1]

# Create two columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Date Selection")
    
    try:
        selected_date_str = st.select_slider(
            "Select date:",
            options=dates,
            value=dates[-1]
        )
        # Convert string date back to datetime
        selected_date_obj = datetime.strptime(selected_date_str, '%m/%d/%Y')
        
        # Get province data first (cached part)
        date_str, province_data = get_province_data(raw_df, selected_date_obj)
        
        # Then get the full geodataframe (non-cached part with geometry)
        if shapefile is not None and date_str is not None and province_data is not None:
            gdf = shapefile.merge(province_data, left_on='NAME_1', right_on='Province', how='left')
            gdf[date_str] = gdf['New_Cases']
        else:
            gdf = None
    except Exception as e:
        st.error(f"Error processing selected date: {e}")
        st.exception(e)
        date_str, province_data, gdf = None, None, None
    
    # Calculate COVID metrics for selected date
    if province_data is not None:
        try:
            # Calculate metrics
            total_cases = int(province_data['New_Cases'].sum())
            max_province = province_data.loc[province_data['New_Cases'].idxmax(), 'Province']
            max_cases = int(province_data['New_Cases'].max())
            
            # Display metrics
            st.metric("Total New Cases", f"{total_cases:,}")
            st.metric("Highest Province", max_province)
            st.metric("Cases in Highest", f"{max_cases:,}")
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")
    
    # Add data table toggle
    show_table = st.checkbox("Show data table", value=False)

# 3. Create map function (no caching as maps can't be pickled)
def make_map(gdf, date):
    # Create map with improved styling
    m = folium.Map(
        location=[-2.5, 118], 
        zoom_start=5, 
        tiles="CartoDB positron",
        prefer_canvas=True
    )
    
    # Add choropleth with better color scheme
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
        highlight=True,
        smooth_factor=0.5
    ).add_to(m)
    
    # Add tooltips with better styling
    folium.GeoJson(
        gdf,
        style_function=lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1},
        highlight_function=lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NAME_1', date],
            aliases=['Province:', 'New cases:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; "
                  "padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)") 
        )
    ).add_to(m)
    
    # Add title to map
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; width: 250px; height: 30px; 
    background-color: rgba(255,255,255,0.8); border-radius: 5px; 
    font-size: 14px; font-weight: bold; text-align: center; 
    line-height: 30px; padding: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)">
    COVID-19 New Cases: {}</div>
    '''.format(date)
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

with col2:
    # Create and display map with subheader
    st.subheader(f"Choropleth for {selected_date_str}")
    
    try:
        if gdf is not None and date_str in gdf.columns:
            with st.spinner(f"Creating map for {selected_date_str}..."):
                m = make_map(gdf, date_str)
                folium_static(m, width=800, height=500)
        else:
            st.error(f"Selected date {selected_date_str} not available in the data")
    except Exception as e:
        st.error(f"Error creating map: {e}")
        st.exception(e)

# Display data table conditionally
if show_table and raw_df is not None:
    st.subheader("Data Table")
    try:
        # Filter data for selected date
        table_data = raw_df[raw_df['Date'] == selected_date_obj][['Location', 'New Cases']].copy()
        table_data = table_data[table_data['New Cases'].notna()].sort_values('New Cases', ascending=False)
        
        # Display with formatting
        st.dataframe(
            table_data,
            column_config={
                "Location": st.column_config.TextColumn("Province"),
                "New Cases": st.column_config.NumberColumn(
                    "New Cases",
                    format="%d"
                )
            },
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.error(f"Error displaying data table: {e}")

# Process trend data more efficiently
@st.cache_data(ttl=3600)
def get_trend_data(raw_df, provinces, max_points=100):
    """Get trend data with optimized point reduction"""
    if not provinces:
        return None
    
    # Filter by provinces
    filtered = raw_df[raw_df['Location'].isin(provinces)].copy()
    
    # If dataset is very large, sample points to improve performance
    dates = filtered['Date'].nunique()
    if dates > max_points:
        # Get evenly distributed dates
        all_dates = sorted(filtered['Date'].unique())
        step = max(1, len(all_dates) // max_points)
        keep_dates = set(all_dates[::step])
        filtered = filtered[filtered['Date'].isin(keep_dates)]
    
    return filtered

# Get default provinces with safe caching
@st.cache_data(ttl=3600)
def get_top_provinces(df):
    """Get top provinces with safe caching"""
    # Group by location and sum cases
    province_totals = df.groupby('Location')['New Cases'].sum().nlargest(3)
    return province_totals.index.tolist()

# Add trend analysis section
st.subheader("COVID-19 Trend Analysis")
if raw_df is not None:
    # Get default provinces
    default_provinces = get_top_provinces(raw_df)
    
    # Create a simple trend chart for selected provinces
    provinces = st.multiselect(
        "Select provinces to compare:",
        options=sorted(raw_df['Location'].unique()),
        default=default_provinces
    )
    
    if provinces:
        # Get trend data with optimization
        trend_data = get_trend_data(raw_df, provinces)
        
        if trend_data is not None:
            fig = px.line(
                trend_data, 
                x='Date', 
                y='New Cases', 
                color='Location',
                title="COVID-19 New Cases Trend",
                labels={"Date": "Date", "New Cases": "New Cases", "Location": "Province"}
            )
            
            fig.update_layout(
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Add footer with information
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #666;">
    <p>Data source: Indonesia Ministry of Health • Last updated: May 2025</p>
</div>
""", unsafe_allow_html=True)

# Add debug and performance section (hidden by default)
with st.expander("Debug & Performance Information", expanded=False):
    st.subheader("Debug Information")
    
    if gdf is not None:
        st.write("GeoDataFrame columns:", gdf.columns.tolist())
        st.write("Selected date column exists:", date_str in gdf.columns)
        st.write("First few rows of GeoDataFrame:")
        st.write(safe_display(gdf.head()))
    
    st.subheader("Performance Metrics")
    st.write("Total number of dates in dataset:", len(dates))
    st.write("Total number of provinces:", raw_df['Location'].nunique())
    st.write("Total data points:", len(raw_df))
    
    # Cache info
    st.write("Cache Info:")
    for key, val in st.cache_data.get_stats().items():
        st.write(f"- {key}: {val}")
