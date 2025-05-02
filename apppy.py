import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from datetime import datetime
import numpy as np
import plotly.express as px

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

# 1. Load data with improved caching for performance
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(csv_path, shapefile_path):
    try:
        # Load CSV more efficiently
        df = pd.read_csv(csv_path, sep=';', parse_dates=['Date'])
        df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
        df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
        
        # Optimize pivot table creation
        locations = df['Location'].unique()
        dates = df['Date'].unique()
        
        # Create a dictionary for faster lookup
        data_dict = {}
        for _, row in df.iterrows():
            data_dict[(row['Date'], row['Location'])] = row['New Cases']
        
        # Create pivot table using dictionary
        pivot_data = []
        for date in dates:
            row_dict = {'Date': date}
            for loc in locations:
                key = (date, loc)
                row_dict[loc] = data_dict.get(key, np.nan)
            pivot_data.append(row_dict)
            
        pivot = pd.DataFrame(pivot_data)
        
        # Convert dates to string format for display
        date_strings = [d.strftime('%m/%d/%Y') for d in dates]
        date_dict = {d: d.strftime('%m/%d/%Y') for d in dates}
        
        # Load shapefile - only read necessary columns for performance
        shp = gpd.read_file(shapefile_path)[['NAME_1', 'geometry']]
        
        # Pre-process shapefile for faster rendering
        # Simplify geometry to reduce complexity (adjust tolerance as needed)
        shp['geometry'] = shp['geometry'].simplify(tolerance=0.01, preserve_topology=True)
        
        # Process for map - optimize by creating columns all at once
        merged_gdf = shp.copy()
        
        # Create a dictionary to map province names to their index in the shapefile
        province_to_idx = {name: i for i, name in enumerate(shp['NAME_1'])}
        
        # Create a date_str to date mapping
        date_str_to_date = {date_str: date for date, date_str in zip(dates, date_strings)}
        
        # Prepare all date columns at once
        for date_str in date_strings:
            merged_gdf[date_str] = np.nan
            
        # Fill in the data - more efficient approach
        for loc in locations:
            if loc in province_to_idx:
                idx = province_to_idx[loc]
                for date_str, date in date_str_to_date.items():
                    key = (date, loc)
                    if key in data_dict:
                        merged_gdf.at[idx, date_str] = data_dict[key]
        
        return date_strings, merged_gdf, pivot, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)
        return [], None, None, None

# Show loading message while data is being prepared
with st.spinner("Loading COVID-19 data..."):
    # Load data
    dates, gdf, data_clean, raw_df = load_data(CSV_PATH, f"{SHAPEFILE_DIR}/IDN_Indonesia_1.shp")

if not dates or gdf is None:
    st.error("Failed to load data. Please check your file paths and data format.")
    st.stop()

# 2. Display date selector with chronological ordering
# Sort dates chronologically
dates_dt = [datetime.strptime(d, '%m/%d/%Y') for d in dates]
dates_sorted_indices = np.argsort(dates_dt)
dates_sorted = [dates[i] for i in dates_sorted_indices]
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
    
    # Calculate COVID metrics for selected date - optimize calculation
    if data_clean is not None:
        try:
            # More efficient metrics calculation
            selected_date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
            date_idx = data_clean.index[data_clean['Date'] == selected_date_obj].tolist()[0]
            date_data = data_clean.iloc[date_idx].drop('Date')
            
            # Calculate metrics efficiently
            total_cases = int(date_data.sum())
            max_province = date_data.idxmax()
            max_cases = int(date_data.max())
            
            # Display metrics
            st.metric("Total New Cases", f"{total_cases:,}")
            st.metric("Highest Province", max_province)
            st.metric("Cases in Highest", f"{max_cases:,}")
        except:
            st.warning("Could not compute metrics for the selected date.")
    
    # Add data table toggle
    show_table = st.checkbox("Show data table", value=False)

# 3. Create enhanced map - optimize map generation
@st.cache_data(ttl=3600)
def prepare_geojson(gdf, date):
    """Pre-process GeoJSON data for better performance"""
    # Create a simplified copy for the choropleth
    gdf_simple = gdf[['NAME_1', 'geometry', date]].copy()
    # Return necessary data structures
    return gdf_simple.__geo_interface__

def make_map(gdf, date):
    # Create map with improved styling
    m = folium.Map(
        location=[-2.5, 118], 
        zoom_start=5, 
        tiles="CartoDB positron",
        prefer_canvas=True
    )
    
    # Use pre-processed GeoJSON
    geo_data = prepare_geojson(gdf, date)
    
    # Add choropleth with better color scheme
    choropleth = folium.Choropleth(
        geo_data=geo_data,
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
    
    # Add tooltips with better styling - optimize by filtering only needed columns
    tooltip_data = gdf[['NAME_1', date, 'geometry']].copy()
    
    folium.GeoJson(
        tooltip_data,
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
    st.subheader(f"Choropleth for {selected_date}")
    
    try:
        if gdf is not None and selected_date in gdf.columns:
            with st.spinner(f"Creating map for {selected_date}..."):
                m = make_map(gdf, selected_date)
                folium_static(m, width=800, height=500)
        else:
            st.error(f"Selected date {selected_date} not available in the data")
            st.write("Available columns:", gdf.columns.tolist() if gdf is not None else "None")
    except Exception as e:
        st.error(f"Error creating map: {e}")
        st.exception(e)
        if gdf is not None:
            st.write("Available columns:", gdf.columns.tolist())
            st.write("Sample data:", safe_display(gdf.head()))

# Display data table conditionally - optimize table creation
if show_table and raw_df is not None:
    st.subheader("Data Table")
    try:
        # More efficient table data filtering
        selected_date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
        
        # Filter directly without using loc
        table_data = raw_df[raw_df['Date'] == selected_date_obj][['Location', 'New Cases']]
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

# Optimize trend analysis section with caching
st.subheader("COVID-19 Trend Analysis")

# Cache the filtered trend data for better performance
@st.cache_data(ttl=3600)
def get_trend_data(raw_df, provinces):
    if not provinces:
        return None
    return raw_df[raw_df['Location'].isin(provinces)]

if raw_df is not None:
    # Get top provinces by case count for default selection
    @st.cache_data(ttl=3600)
    def get_top_provinces(df, n=3):
        return df['Location'].value_counts().nlargest(n).index.tolist()
    
    default_provinces = get_top_provinces(raw_df)
    
    # Create a simple trend chart for selected provinces
    provinces = st.multiselect(
        "Select provinces to compare:",
        options=sorted(raw_df['Location'].unique()),
        default=default_provinces
    )
    
    trend_data = get_trend_data(raw_df, provinces)
    
    if trend_data is not None and not trend_data.empty:
        # Create plotly figure
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

# Add debug section (hidden by default)
with st.expander("Debug Information", expanded=False):
    st.subheader("Debug Information")
    if gdf is not None:
        st.write("GeoDataFrame columns:", gdf.columns.tolist())
        st.write("Selected date column exists:", selected_date in gdf.columns)
        st.write("First few rows of GeoDataFrame:")
        st.write(safe_display(gdf.head()))
    else:
        st.error("GeoDataFrame is None")
