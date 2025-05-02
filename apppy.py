import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from datetime import datetime
import plotly.express as px
import os
import json
import time

# --- CONFIG ---
CSV_PATH = 'data/indonesia_data.csv'
SHAPEFILE_DIR = 'data'
CACHE_DIR = 'cache'

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

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Title
st.title("Indonesia COVID-19 New Cases Choropleth Map")

# --- SUPER FAST DATA LOADING STRATEGY ---

@st.cache_data(ttl=3600)
def load_and_prepare_base_data(csv_path):
    """
    Load CSV data and pre-process into dates and basic statistics
    """
    start_time = time.time()
    
    try:
        # Load CSV with direct date parsing
        df = pd.read_csv(csv_path, sep=';', parse_dates=['Date'])
        df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
        df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
        
        # Get unique dates in chronological order
        all_dates = sorted(df['Date'].unique())
        date_strings = [d.strftime('%m/%d/%Y') for d in all_dates]
        
        # Pre-compute metrics for all dates
        date_metrics = {}
        for date in all_dates:
            date_str = date.strftime('%m/%d/%Y')
            date_data = df[df['Date'] == date]
            
            if not date_data.empty:
                total_cases = int(date_data['New Cases'].sum())
                max_idx = date_data['New Cases'].idxmax() if not date_data['New Cases'].isna().all() else None
                max_province = date_data.loc[max_idx, 'Location'] if max_idx is not None else "Unknown"
                max_cases = int(date_data['New Cases'].max()) if max_idx is not None else 0
                
                date_metrics[date_str] = {
                    'total_cases': total_cases,
                    'max_province': max_province,
                    'max_cases': max_cases
                }
        
        # Pre-compute trend data for top provinces
        province_totals = df.groupby('Location')['New Cases'].sum().nlargest(5)
        top_provinces = province_totals.index.tolist()
        
        # Calculate trend data for top provinces
        trend_data = {}
        for province in top_provinces:
            province_data = df[df['Location'] == province].copy()
            if len(province_data) > 100:  # If too many points, sample
                province_data = province_data.iloc[::len(province_data)//100 + 1]
            trend_data[province] = {
                'dates': [d.strftime('%Y-%m-%d') for d in province_data['Date']],
                'values': province_data['New Cases'].fillna(0).tolist()
            }
        
        print(f"Data loading took {time.time() - start_time:.2f} seconds")
        return df, all_dates, date_strings, date_metrics, top_provinces, trend_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)
        return None, [], [], {}, [], {}

@st.cache_data(ttl=3600)
def prepare_geojson():
    """
    Prepare simplified GeoJSON for mapping
    """
    # Look for pre-processed GeoJSON first (much faster than shapefile)
    geojson_path = f"{CACHE_DIR}/indonesia_simple.geojson"
    
    if os.path.exists(geojson_path):
        try:
            with open(geojson_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load cached GeoJSON: {e}")
    
    # If no cached file, try to load from shapefile
    try:
        import geopandas as gpd
        
        # Try different possible shapefile paths
        shapefile_options = [
            f"{SHAPEFILE_DIR}/IDN_Indonesia_1.shp",
            f"{SHAPEFILE_DIR}/IDN_adm1.shp",
            f"{SHAPEFILE_DIR}/indonesia.shp",
            f"{SHAPEFILE_DIR}/IDN.shp"
        ]
        
        shapefile_path = None
        for option in shapefile_options:
            if os.path.exists(option):
                shapefile_path = option
                break
        
        if shapefile_path is None:
            st.error("Could not find any suitable shapefile")
            return None
            
        # Load and simplify
        shp = gpd.read_file(shapefile_path)
        
        # Simplify to reduce file size (tolerance controls simplification level)
        shp_simple = shp.simplify(tolerance=0.01)
        
        # Convert to GeoJSON
        geojson_data = json.loads(shp_simple.to_json())
        
        # Cache for future use
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f)
            
        return geojson_data
    except Exception as e:
        st.error(f"Could not process shapefile: {e}")
        st.exception(e)
        return None

# Load data with a spinner
with st.spinner("Loading COVID-19 data..."):
    raw_df, all_dates, date_strings, date_metrics, top_provinces, trend_data = load_and_prepare_base_data(CSV_PATH)
    geojson_data = prepare_geojson()

if raw_df is None or geojson_data is None:
    st.error("Failed to load required data. Please check your data files.")
    st.stop()

# Function to get data for a specific date (very fast)
def get_date_data(raw_df, date_obj):
    """Get data for specific date with minimal processing"""
    date_str = date_obj.strftime('%m/%d/%Y')
    date_data = raw_df[raw_df['Date'] == date_obj].copy()
    return date_str, date_data

# Create two columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Date Selection")
    
    selected_date_str = st.select_slider(
        "Select date:",
        options=date_strings,
        value=date_strings[-1]
    )
    
    # Display pre-computed metrics
    if selected_date_str in date_metrics:
        metrics = date_metrics[selected_date_str]
        st.metric("Total New Cases", f"{metrics['total_cases']:,}")
        st.metric("Highest Province", metrics['max_province'])
        st.metric("Cases in Highest", f"{metrics['max_cases']:,}")
    
    # Add data table toggle
    show_table = st.checkbox("Show data table", value=False)

# Create map function
def make_fast_map(geojson_data, date_data, date_str):
    """Create map with optimized processing"""
    # Create province-data dictionary for this date
    province_data = {}
    
    for _, row in date_data.iterrows():
        province_data[row['Location']] = row['New Cases']
    
    # Create base map
    m = folium.Map(
        location=[-2.5, 118], 
        zoom_start=5, 
        tiles="CartoDB positron",
        prefer_canvas=True
    )
    
    # Add choropleth
    folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=province_data,
        columns=["Province", "Value"],  # Not actually used with data_dict
        key_on='feature.properties.NAME_1',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"New COVID-19 Cases ({date_str})",
        highlight=True,
        smooth_factor=0.5,
        line_color='#ffffff',
        line_weight=0.5,
        data_dict=province_data  # Direct data dictionary is much faster
    ).add_to(m)
    
    # Add title to map
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; width: 250px; height: 30px; 
    background-color: rgba(255,255,255,0.8); border-radius: 5px; 
    font-size: 14px; font-weight: bold; text-align: center; 
    line-height: 30px; padding: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)">
    COVID-19 New Cases: {}</div>
    '''.format(date_str)
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

with col2:
    # Create and display map with subheader
    st.subheader(f"Choropleth for {selected_date_str}")
    
    try:
        # Convert string date back to datetime
        selected_date_obj = datetime.strptime(selected_date_str, '%m/%d/%Y')
        
        # Get data for this date (fast)
        date_str, date_data = get_date_data(raw_df, selected_date_obj)
        
        with st.spinner(f"Creating map for {selected_date_str}..."):
            m = make_fast_map(geojson_data, date_data, date_str)
            folium_static(m, width=800, height=500)
    except Exception as e:
        st.error(f"Error creating map: {e}")
        st.exception(e)

# Display data table conditionally
if show_table and raw_df is not None:
    st.subheader("Data Table")
    try:
        # Convert selected_date_str to datetime
        selected_date_obj = datetime.strptime(selected_date_str, '%m/%d/%Y')
        
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

# Add trend analysis section with pre-computed data
st.subheader("COVID-19 Trend Analysis")

if trend_data:
    # Create a simple trend chart for selected provinces
    provinces = st.multiselect(
        "Select provinces to compare:",
        options=sorted(raw_df['Location'].unique()),
        default=top_provinces[:3]
    )
    
    if provinces:
        # Create plot data from pre-computed and on-demand data
        plot_data = []
        
        for province in provinces:
            if province in trend_data:
                # Use pre-computed data for top provinces
                dates = [datetime.strptime(d, '%Y-%m-%d') for d in trend_data[province]['dates']]
                values = trend_data[province]['values']
                
                for date, value in zip(dates, values):
                    plot_data.append({
                        'Date': date,
                        'New Cases': value,
                        'Location': province
                    })
            else:
                # Calculate on-demand for other provinces (with sampling)
                province_data = raw_df[raw_df['Location'] == province].copy()
                if len(province_data) > 100:
                    province_data = province_data.iloc[::len(province_data)//100 + 1]
                
                for _, row in province_data.iterrows():
                    plot_data.append({
                        'Date': row['Date'],
                        'New Cases': row['New Cases'],
                        'Location': province
                    })
        
        if plot_data:
            # Convert to DataFrame
            trend_df = pd.DataFrame(plot_data)
            
            fig = px.line(
                trend_df, 
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
    st.subheader("Performance Metrics")
    st.write("Total number of dates in dataset:", len(date_strings))
    st.write("Total number of provinces:", raw_df['Location'].nunique())
    st.write("Total data points:", len(raw_df))
    
    # Cache info
    st.write("Cache Info:")
    for key, val in st.cache_data.get_stats().items():
        st.write(f"- {key}: {val}")
        
    # Additional debug info
    if st.checkbox("Show raw data sample"):
        st.write("Raw data sample (first 5 rows):")
        st.dataframe(raw_df.head())
    
    if st.checkbox("Show pre-computed metrics"):
        st.write("Pre-computed metrics for first 3 dates:")
        st.json({k: date_metrics[k] for k in list(date_metrics.keys())[:3]})
