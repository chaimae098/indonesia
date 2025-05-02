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
CSV_PATH = 'data/indonesia_data.csv'  # Updated path to match first code example
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
    /* Card styling for map */
    .map-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
# If data directory doesn't exist, create it
os.makedirs(SHAPEFILE_DIR, exist_ok=True)

# Title
st.title("Indonesia COVID-19 New Cases Choropleth Map")

# --- FAST DATA LOADING STRATEGY ---

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
        province_totals = df.groupby('Location')['New Cases'].sum().nlargest(10)
        top_provinces = province_totals.index.tolist()
        
        # Calculate trend data for top provinces
        trend_data = {}
        for province in top_provinces:
            province_data = df[df['Location'] == province].copy()
            # If too many points, sample to improve performance
            if len(province_data) > 50:
                province_data = province_data.iloc[::len(province_data)//50 + 1]
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
    Prepare simplified GeoJSON for mapping - using cached version when possible
    """
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Look for pre-processed GeoJSON first (much faster than shapefile)
    geojson_path = f"{CACHE_DIR}/indonesia_simple.geojson"
    
    if os.path.exists(geojson_path):
        try:
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
                # Inspect structure to identify available properties
                if 'features' in geojson_data and len(geojson_data['features']) > 0:
                    sample_feature = geojson_data['features'][0]
                    available_properties = sample_feature.get('properties', {}).keys()
                    st.session_state['geojson_properties'] = list(available_properties)
                return geojson_data
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
        
        # Inspect the shapefile's columns to identify province name column
        st.session_state['shapefile_columns'] = list(shp.columns)
        
        # Simplify to reduce file size (tolerance controls simplification level)
        shp_simple = shp.simplify(tolerance=0.01)
        
        # Convert to GeoJSON
        geojson_data = json.loads(shp_simple.to_json())
        
        # Store available properties
        if 'features' in geojson_data and len(geojson_data['features']) > 0:
            sample_feature = geojson_data['features'][0]
            available_properties = sample_feature.get('properties', {}).keys()
            st.session_state['geojson_properties'] = list(available_properties)
        
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
    # Check if data file exists first
    if not os.path.exists(CSV_PATH):
        st.error(f"Data file not found: {CSV_PATH}")
        st.info("Please make sure your data is in the correct location. The app expects data in: data/indonesia_data.csv")
        
        # Create a sample data file for demo purposes if it doesn't exist
        try:
            os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
            
            # Check if we should create sample data
            create_sample = st.button("Create sample data file for demonstration")
            
            if create_sample:
                # Create minimal sample data
                st.info("Creating sample data file...")
                
                # Generate sample data
                import numpy as np
                from datetime import datetime, timedelta
                
                # Create dates
                base_date = datetime(2023, 1, 1)
                dates = [base_date + timedelta(days=i) for i in range(30)]
                
                # Create provinces
                provinces = ['Java', 'Sumatra', 'Sulawesi', 'Kalimantan', 'Papua']
                
                # Create DataFrame
                rows = []
                for date in dates:
                    for province in provinces:
                        # Random cases between 0 and 100
                        cases = int(np.random.randint(0, 100))
                        rows.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Location': province,
                            'New Cases': cases
                        })
                
                sample_df = pd.DataFrame(rows)
                
                # Save to CSV
                sample_df.to_csv(CSV_PATH, sep=';', index=False)
                st.success(f"Sample data created at {CSV_PATH}")
                st.info("Refresh the page to use the sample data")
        except Exception as e:
            st.error(f"Could not create sample data: {e}")
        
        st.stop()
        
    raw_df, all_dates, date_strings, date_metrics, top_provinces, trend_data = load_and_prepare_base_data(CSV_PATH)
    geojson_data = prepare_geojson()

if raw_df is None or geojson_data is None:
    st.error("Failed to load required data. Please check your data files.")
    
    # Show more helpful error details
    if raw_df is None:
        st.info(f"The CSV file was found but could not be processed. Make sure it has the correct columns (Date, Location, New Cases) and separator (;).")
    
    if geojson_data is None:
        st.info(f"Could not load GeoJSON data. Make sure the shapefile exists in the {SHAPEFILE_DIR} directory.")
        
        # Check which shapefiles exist
        shapefile_options = [
            f"{SHAPEFILE_DIR}/IDN_Indonesia_1.shp",
            f"{SHAPEFILE_DIR}/IDN_adm1.shp",
            f"{SHAPEFILE_DIR}/indonesia.shp",
            f"{SHAPEFILE_DIR}/IDN.shp"
        ]
        
        existing_files = [f for f in shapefile_options if os.path.exists(f)]
        if existing_files:
            st.info(f"Found these shapefiles: {', '.join(existing_files)}")
        else:
            st.info(f"No shapefiles found in {SHAPEFILE_DIR}. Please add the shapefile to this directory.")
    
    st.stop()

# Determine the right property key for provinces in GeoJSON
if 'geojson_properties' not in st.session_state and geojson_data:
    # Inspect structure to identify available properties
    if 'features' in geojson_data and len(geojson_data['features']) > 0:
        sample_feature = geojson_data['features'][0]
        available_properties = sample_feature.get('properties', {}).keys()
        st.session_state['geojson_properties'] = list(available_properties)

# Function to get data for a specific date (very fast)
def get_date_data(raw_df, date_obj):
    """Get data for specific date with minimal processing"""
    date_data = raw_df[raw_df['Date'] == date_obj].copy()
    return date_data

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
    
    # Add GeoJSON property selection
    if 'geojson_properties' in st.session_state:
        property_options = st.session_state['geojson_properties']
        
        # Try to find the best property key for provinces
        default_property = None
        priority_names = ['NAME_1', 'name', 'province', 'PROVINCE', 'name_1', 'NAME', 'ADMIN']
        
        for name in priority_names:
            if name in property_options:
                default_property = name
                break
                
        if default_property is None and property_options:
            default_property = property_options[0]
            
        selected_property = st.selectbox(
            "Select GeoJSON property for provinces:",
            options=property_options,
            index=property_options.index(default_property) if default_property in property_options else 0
        )
        
        # Store the selected property
        st.session_state['selected_property'] = selected_property
    else:
        # Fallback to default property
        st.session_state['selected_property'] = 'NAME_1'

# Create map function
def make_fast_map(geojson_data, date_data):
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
        prefer_canvas=True  # Use Canvas renderer for better performance
    )
    
    # Use the correct property key from session state
    property_key = st.session_state.get('selected_property', 'NAME_1')
    key_on = f'feature.properties.{property_key}'
    
    # Add choropleth
    try:
        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            data=province_data,
            columns=["Province", "Value"],  # Not actually used with data_dict
            key_on=key_on,
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"New COVID-19 Cases",
            highlight=True,
            smooth_factor=0.5,  # Smoother lines for better performance
            line_color='#ffffff',
            line_weight=0.5,
            data_dict=province_data  # Direct data dictionary is much faster
        ).add_to(m)
        
        # Add tooltips
        folium.GeoJson(
            geojson_data,
            name='labels',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'transparent'
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[property_key],
                aliases=['Province:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
            )
        ).add_to(m)
        
    except Exception as e:
        st.error(f"Error creating choropleth with property '{property_key}': {e}")
        
        # Fallback to just displaying the base map with borders
        folium.GeoJson(
            geojson_data,
            name='geojson',
            style_function=lambda x: {
                'fillColor': '#ffff00',
                'color': '#000000',
                'fillOpacity': 0.1,
                'weight': 0.5
            }
        ).add_to(m)
    
    # Add title to map
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; width: 250px; height: 30px; 
    background-color: rgba(255,255,255,0.8); border-radius: 5px; 
    font-size: 14px; font-weight: bold; text-align: center; 
    line-height: 30px; padding: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)">
    COVID-19 New Cases: {}</div>
    '''.format(selected_date_str)
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

with col2:
    # Create and display map with subheader
    st.subheader(f"COVID-19 Map for {selected_date_str}")
    
    # Show property key info
    if 'selected_property' in st.session_state:
        st.caption(f"Using GeoJSON property: '{st.session_state['selected_property']}' to map provinces")
    
    try:
        # Convert string date back to datetime
        selected_date_obj = datetime.strptime(selected_date_str, '%m/%d/%Y')
        
        # Get data for this date (fast)
        date_data = get_date_data(raw_df, selected_date_obj)
        
        # Only create map if there's data
        if not date_data.empty:
            with st.spinner(f"Creating map for {selected_date_str}..."):
                m = make_fast_map(geojson_data, date_data)
                folium_static(m, width=800, height=500)
        else:
            st.warning(f"No data available for {selected_date_str}")
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
        default=top_provinces[:3] if top_provinces else []
    )
    
    if provinces:
        # Create plot data from pre-computed and on-demand data
        plot_data = []
        
        for province in provinces:
            if province in trend_data:
                # Use pre-computed data for top provinces (faster)
                dates = [datetime.strptime(d, '%Y-%m-%d') for d in trend_data[province]['dates']]
                values = trend_data[province]['values']
                
                for date, value in zip(dates, values):
                    plot_data.append({
                        'Date': date,
                        'New Cases': value,
                        'Location': province
                    })
            else:
                # Calculate on-demand for other provinces (with sampling for performance)
                province_data = raw_df[raw_df['Location'] == province].copy()
                if len(province_data) > 50:  # Sample data for better performance
                    province_data = province_data.iloc[::len(province_data)//50 + 1]
                
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
    <p>Created with Streamlit and Folium</p>
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
    
    # Show GeoJSON Properties
    if 'geojson_properties' in st.session_state:
        st.write("Available GeoJSON Properties:")
        st.write(st.session_state['geojson_properties'])
    
    # Show Shapefile Columns
    if 'shapefile_columns' in st.session_state:
        st.write("Shapefile Columns:")
        st.write(st.session_state['shapefile_columns'])
        
    # Additional debug info
    if st.checkbox("Show raw data sample"):
        st.write("Raw data sample (first 5 rows):")
        st.dataframe(raw_df.head())
