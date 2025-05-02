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

# Apply custom CSS for better aesthetics - moved to external CSS to improve loading time
st.markdown("""
<style>
    .main {background-color: #f5f7f9;}
    h1 {color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; padding-bottom: 20px;}
    h2, h3 {color: #2563EB; font-family: 'Helvetica Neue', sans-serif;}
    .stSlider > div > div > div {background-color: #EF4444;}
    .stSlider [data-baseweb="slider"] {height: 6px;}
    div[data-testid="stExpander"] p {font-size: 16px;}
    .map-container {background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Indonesia COVID-19 New Cases Choropleth Map")

# Improved data loading with optimized caching strategy
@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(csv_path):
    """Load and preprocess CSV data efficiently"""
    df = pd.read_csv(csv_path, sep=';')
    df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
    df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_shapefile(shapefile_path):
    """Load shapefile separately to optimize memory usage"""
    return gpd.read_file(shapefile_path)

@st.cache_data(ttl=3600, show_spinner=False)
def process_data(df):
    """Create pivot table and date strings once"""
    pivot = df.pivot(index='Date', columns='Location', values='New Cases').reset_index()
    date_strings = pivot['Date'].dt.strftime('%m/%d/%Y').tolist()
    return pivot, date_strings

@st.cache_data(ttl=3600, show_spinner=False)
def create_choropleth_data(pivot, shp):
    """Pre-compute merged geodataframe for all dates"""
    # Create a dictionary to store each date's data
    date_data_dict = {}
    
    # Process each date once
    for date in pivot['Date']:
        date_str = date.strftime('%m/%d/%Y')
        date_data = pivot.set_index('Date').loc[date].reset_index()
        date_data.columns = ['Province', 'New_Cases']
        date_data_dict[date_str] = date_data
    
    # Merge with shapefile once
    merged_gdf = shp.copy()
    
    # Add data for each date
    for date_str, date_data in date_data_dict.items():
        # Use efficient merge
        province_to_cases = dict(zip(date_data['Province'], date_data['New_Cases']))
        merged_gdf[date_str] = merged_gdf['NAME_1'].map(province_to_cases)
    
    return merged_gdf

# Load data in stages to improve memory usage
with st.spinner("Loading COVID-19 data..."):
    # Load raw data
    raw_df = load_csv(CSV_PATH)
    shp = load_shapefile(f"{SHAPEFILE_DIR}/IDN_Indonesia_1.shp")
    
    # Process data
    data_pivot, dates = process_data(raw_df)
    gdf = create_choropleth_data(data_pivot, shp)

if raw_df.empty or gdf is None:
    st.error("Failed to load data. Please check your file paths and data format.")
    st.stop()

# Sort dates chronologically
dates_dt = [datetime.strptime(d, '%m/%d/%Y') for d in dates]
dates_sorted = [d.strftime('%m/%d/%Y') for d in sorted(dates_dt)]
min_date = dates_sorted[0]
max_date = dates_sorted[-1]

# Create two columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Date Selection")
    selected_date = st.select_slider(
        "Select date:",
        options=dates_sorted,
        value=dates_sorted[-1]
    )
    
    # Calculate COVID metrics for selected date - optimized
    date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
    date_data = data_pivot.set_index('Date').loc[date_obj]
    
    total_cases = int(date_data.sum())
    max_province = date_data.idxmax()
    max_cases = int(date_data.max())
    
    # Display metrics
    st.metric("Total New Cases", f"{total_cases:,}")
    st.metric("Highest Province", max_province)
    st.metric("Cases in Highest", f"{max_cases:,}")
    
    # Add data table toggle
    show_table = st.checkbox("Show data table", value=False)

# Create map with simplified GeoJSON for better performance
@st.cache_data(ttl=3600)
def make_lightweight_map(gdf, date):
    """Create a more efficient map by simplifying geometries"""
    # Create base map
    m = folium.Map(
        location=[-2.5, 118], 
        zoom_start=5, 
        tiles="CartoDB positron",
        prefer_canvas=True
    )
    
    # Simplify geometries for better performance
    simplified_gdf = gdf.copy()
    if hasattr(simplified_gdf.geometry.iloc[0], 'simplify'):
        simplified_gdf['geometry'] = simplified_gdf['geometry'].simplify(0.01)
    
    # Convert to geojson only once
    geo_data = simplified_gdf.__geo_interface__
    
    # Add choropleth
    choropleth = folium.Choropleth(
        geo_data=geo_data,
        name="choropleth",
        data=simplified_gdf,
        columns=['NAME_1', date],
        key_on='feature.properties.NAME_1',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"New COVID-19 Cases ({date})",
        highlight=True,
        smooth_factor=1.0  # Increased for better performance
    ).add_to(m)
    
    # Add tooltips with optimized styling
    folium.GeoJson(
        simplified_gdf[['NAME_1', date, 'geometry']],  # Only include needed columns
        style_function=lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1},
        highlight_function=lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NAME_1', date],
            aliases=['Province:', 'New cases:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; "
                  "padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)") 
        )
    ).add_to(m)
    
    # Add simplified title to map
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; width: 250px; height: 30px; 
    background-color: rgba(255,255,255,0.8); border-radius: 5px; 
    font-size: 14px; font-weight: bold; text-align: center; 
    line-height: 30px; padding: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)">
    COVID-19 New Cases: {date}</div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

with col2:
    # Create and display map
    st.subheader(f"Choropleth for {selected_date}")
    
    if selected_date in gdf.columns:
        with st.spinner(f"Creating map for {selected_date}..."):
            m = make_lightweight_map(gdf, selected_date)
            folium_static(m, width=800, height=500)
    else:
        st.error(f"Selected date {selected_date} not available in the data")

# Display data table conditionally - optimized query
if show_table:
    st.subheader("Data Table")
    # Get data for selected date efficiently
    date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
    table_data = data_pivot.set_index('Date').loc[date_obj].sort_values(ascending=False).reset_index()
    table_data.columns = ['Province', 'New Cases']
    table_data = table_data[table_data['New Cases'].notna()]
    
    # Use a more efficient table display
    st.dataframe(
        table_data,
        column_config={
            "Province": st.column_config.TextColumn("Province"),
            "New Cases": st.column_config.NumberColumn("New Cases", format="%d")
        },
        use_container_width=True,
        hide_index=True
    )

# Optimized trend analysis section - using precomputed data
st.subheader("COVID-19 Trend Analysis")

# Cache province list to avoid recalculation
@st.cache_data(ttl=3600)
def get_top_provinces(df, n=3):
    return df['Location'].value_counts().nlargest(n).index.tolist()

top_provinces = get_top_provinces(raw_df)
provinces = st.multiselect(
    "Select provinces to compare:",
    options=sorted(raw_df['Location'].unique()),
    default=top_provinces
)

# Efficient trend data filtering
if provinces:
    # Pre-filter data for selected provinces
    @st.cache_data(ttl=3600)
    def get_trend_data(df, selected_provinces):
        filtered_data = df[df['Location'].isin(selected_provinces)].copy()
        return filtered_data
    
    trend_data = get_trend_data(raw_df, provinces)
    
    # Create plotly chart with optimized parameters
    fig = px.line(
        trend_data, 
        x='Date', 
        y='New Cases', 
        color='Location',
        title="COVID-19 New Cases Trend",
        labels={"Date": "Date", "New Cases": "New Cases", "Location": "Province"}
    )
    
    # Optimize plotly render performance
    fig.update_layout(
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )
    
    # Set decimals to 0 for better performance and readability
    fig.update_yaxes(tickformat=",.0f")
    
    st.plotly_chart(fig, use_container_width=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #666;">
    <p>Data source: Indonesia Ministry of Health • Last updated: May 2025</p>
</div>
""", unsafe_allow_html=True)

# Add debug section (only create if expanded to save resources)
with st.expander("Debug Information", expanded=False):
    st.subheader("Debug Information")
    if st.checkbox("Show detailed debug info", value=False):
        st.write("GeoDataFrame columns:", gdf.columns.tolist())
        st.write("Selected date column exists:", selected_date in gdf.columns)
        st.write("Memory usage:")
        st.write(f"- Raw dataframe: {raw_df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        st.write(f"- GeoDataFrame: {gdf.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
