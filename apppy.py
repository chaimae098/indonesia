import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from datetime import datetime
import plotly.express as px
import os
import numpy as np
import gc  # For garbage collection

# --- CONFIG ---
CSV_PATH = 'data/indonesia_data.csv'
SHAPEFILE_DIR = 'data'

# Set page configuration - optimized for performance
st.set_page_config(
    page_title="Indonesia COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start collapsed for faster initial load
)

# Minimal CSS for better load time
st.markdown("""
<style>
.main {background-color: #f5f7f9;}
h1 {color: #1E3A8A;}
h2, h3 {color: #2563EB;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Indonesia COVID-19 New Cases Choropleth Map")

# Improved data loading with optimized caching strategy and better error handling
@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(csv_path):
    """Load and preprocess CSV data efficiently"""
    try:
        df = pd.read_csv(csv_path, sep=';')
        df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
        df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

@st.cache_data(ttl=3600, show_spinner=False)
def load_shapefile(shapefile_path):
    """Load shapefile separately to optimize memory usage"""
    try:
        return gpd.read_file(shapefile_path)
    except Exception as e:
        st.error(f"Error loading shapefile: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def process_data(df):
    """Create pivot table and date strings once"""
    try:
        if df.empty:
            return pd.DataFrame(), []
        
        pivot = df.pivot(index='Date', columns='Location', values='New Cases').reset_index()
        date_strings = pivot['Date'].dt.strftime('%m/%d/%Y').tolist()
        return pivot, date_strings
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame(), []

# Split the choropleth data creation to avoid unhashable parameter error
def create_choropleth_data(pivot, shp):
    """Pre-compute merged geodataframe for all dates - without caching"""
    # Create a dictionary to store province-to-cases mapping for each date
    date_mappings = {}
    
    # Process each date once
    for date in pivot['Date']:
        date_str = date.strftime('%m/%d/%Y')
        date_data = pivot.set_index('Date').loc[date].reset_index()
        date_data.columns = ['Province', 'New_Cases']
        # Store mapping as a simple dictionary
        province_to_cases = dict(zip(date_data['Province'], date_data['New_Cases']))
        date_mappings[date_str] = province_to_cases
    
    # Merge with shapefile once
    merged_gdf = shp.copy()
    
    # Add data for each date
    for date_str, mapping in date_mappings.items():
        merged_gdf[date_str] = merged_gdf['NAME_1'].map(mapping)
    
    return merged_gdf

# Load data in stages to improve memory usage
with st.spinner("Loading COVID-19 data..."):
    try:
        # Load raw data first
        raw_df = load_csv(CSV_PATH)
        
        # Only proceed if CSV loaded correctly
        if raw_df.empty:
            st.error("CSV data is empty. Please check your data file.")
            st.stop()
            
        # Load shapefile second
        shp = load_shapefile(f"{SHAPEFILE_DIR}/IDN_Indonesia_1.shp")
        
        # Process data
        data_pivot, dates = process_data(raw_df)
        
        # Create choropleth data after all previous steps succeed
        gdf = create_choropleth_data(data_pivot, shp)
        
        # Quick verification
        if gdf.empty:
            st.error("Failed to create map data.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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
    
    # Handle empty dates list
    if not dates_sorted:
        st.error("No dates available in the data")
        selected_date = ""
    else:
        selected_date = st.select_slider(
            "Select date:",
            options=dates_sorted,
            value=dates_sorted[-1] if dates_sorted else ""
        )
    
    # Calculate COVID metrics for selected date - with error handling
    try:
        if selected_date and not data_pivot.empty:
            date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
            
            # Safely access data for the selected date
            if date_obj in data_pivot['Date'].values:
                date_data = data_pivot.set_index('Date').loc[date_obj]
                
                # Calculate metrics with safety checks
                total_cases = int(date_data.sum()) if not date_data.empty else 0
                
                if not date_data.empty:
                    max_province = date_data.idxmax()
                    max_cases = int(date_data.max())
                else:
                    max_province = "N/A"
                    max_cases = 0
                
                # Display metrics
                st.metric("Total New Cases", f"{total_cases:,}")
                st.metric("Highest Province", max_province)
                st.metric("Cases in Highest", f"{max_cases:,}")
            else:
                st.warning(f"No data available for {selected_date}")
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
    
    # Add data table toggle
    show_table = st.checkbox("Show data table", value=False)

# Create map with simplified GeoJSON for better performance - without caching to avoid unhashable type errors
def make_lightweight_map(gdf, date):
    """Create a more efficient map by simplifying geometries"""
    # Create base map with minimal settings for better performance
    m = folium.Map(
        location=[-2.5, 118], 
        zoom_start=5, 
        tiles="CartoDB positron",
        prefer_canvas=True
    )
    
    # Work with a copy to avoid modifying the original
    data_for_map = gdf[['NAME_1', date, 'geometry']].copy()
    
    # Simplify geometries for better performance
    if hasattr(data_for_map.geometry.iloc[0], 'simplify'):
        data_for_map['geometry'] = data_for_map['geometry'].simplify(0.01)
    
    # Add choropleth - keeping only essential parameters
    choropleth = folium.Choropleth(
        geo_data=data_for_map.__geo_interface__,
        name="choropleth",
        data=data_for_map,
        columns=['NAME_1', date],
        key_on='feature.properties.NAME_1',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"New COVID-19 Cases ({date})",
        smooth_factor=1.0  # Higher for better performance
    ).add_to(m)
    
    # Add tooltips with minimal styling for better performance
    folium.GeoJson(
        data_for_map,
        style_function=lambda x: {'weight': 0.1},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NAME_1', date],
            aliases=['Province:', 'New cases:'],
            style="background-color: white; font-size: 12px; padding: 5px;"
        )
    ).add_to(m)
    
    # Add simplified title to map
    title_html = f'<div style="position:fixed;top:10px;left:50px;background-color:white;padding:5px;font-size:14px;">COVID-19 New Cases: {date}</div>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

with col2:
    # Create and display map
    st.subheader(f"Choropleth for {selected_date}")
    
    if gdf is not None and selected_date in gdf.columns:
        try:
            with st.spinner(f"Creating map for {selected_date}..."):
                m = make_lightweight_map(gdf, selected_date)
                folium_static(m, width=800, height=500)
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            st.info("Try selecting a different date or refreshing the page.")
    else:
        st.error(f"Selected date {selected_date} not available in the data")

# Display data table conditionally - optimized query
if show_table:
    st.subheader("Data Table")
    try:
        # Get data for selected date efficiently
        date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
        
        # Safely access the data
        if date_obj in data_pivot['Date'].values:
            table_data = data_pivot.set_index('Date').loc[date_obj].sort_values(ascending=False).reset_index()
            table_data.columns = ['Province', 'New Cases']
            table_data = table_data[table_data['New Cases'].notna()]
            
            # Use a more efficient table display - limiting to top 20 rows for performance
            st.dataframe(
                table_data.head(20),  # Limit rows for better performance
                column_config={
                    "Province": st.column_config.TextColumn("Province"),
                    "New Cases": st.column_config.NumberColumn("New Cases", format="%d")
                },
                use_container_width=True,
                hide_index=True
            )
            
            if len(table_data) > 20:
                st.info(f"Showing top 20 out of {len(table_data)} provinces for better performance.")
        else:
            st.warning(f"No data available for {selected_date}")
    except Exception as e:
        st.error(f"Error displaying table: {str(e)}")

# Define trend data filtering function before using it
@st.cache_data(ttl=3600)
def get_trend_data(df, province_list):
    """Filter data for specific provinces using simple list of strings"""
    return df[df['Location'].isin(province_list)].copy()

# Cache province list to avoid recalculation
@st.cache_data(ttl=3600)
def get_top_provinces(df, n=3):
    """Get top provinces by case count"""
    return list(df['Location'].value_counts().nlargest(n).index)

# Optimized trend analysis section - using precomputed data
show_trends = st.checkbox("Show COVID-19 Trend Analysis", value=True)

if show_trends:
    st.subheader("COVID-19 Trend Analysis")

    # Only compute if checkbox is checked
    top_provinces = get_top_provinces(raw_df)
    all_provinces = sorted(raw_df['Location'].unique())
    
    provinces = st.multiselect(
        "Select provinces to compare:",
        options=all_provinces,
        default=top_provinces
    )

    # Only render chart if provinces are selected
    if provinces:
        # Get filtered data
        trend_data = get_trend_data(raw_df, provinces)
        
        # Create minimal plotly chart with optimized parameters
        fig = px.line(
            trend_data, 
            x='Date', 
            y='New Cases', 
            color='Location',
            title="COVID-19 New Cases Trend"
        )
        
        # Optimize plotly render performance - minimal settings
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Reduce points for smoother rendering if large dataset
        if len(trend_data) > 1000:
            fig.update_traces(simplify=True)
        
        # Set decimals to 0 for better performance
        fig.update_yaxes(tickformat=",.0f")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Force garbage collection after chart render
        gc.collect()

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
