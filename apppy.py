import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from datetime import datetime
import plotly.express as px
import os
import time

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

# Ensure directories exist
os.makedirs(SHAPEFILE_DIR, exist_ok=True)

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
</style>
""", unsafe_allow_html=True)

# Title
st.title("Indonesia COVID-19 New Cases Choropleth Map")

# --- OPTIMIZED DATA LOADING ---

@st.cache_data(ttl=3600)
def load_data(csv_path):
    """Load and preprocess COVID data"""
    start_time = time.time()
    
    try:
        # Load CSV efficiently with proper data types
        df = pd.read_csv(csv_path, sep=';')
        
        # Convert date to datetime only if needed
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
            
        # Convert cases to numeric and filter out national data
        df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
        df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
        
        # Pre-compute date strings for UI
        all_dates = sorted(df['Date'].unique())
        date_strings = [d.strftime('%Y-%m-%d') for d in all_dates]
        
        # Pre-compute basic statistics per date to avoid recomputation
        date_metrics = {}
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            date_data = df[df['Date'] == date]
            
            if not date_data.empty:
                date_metrics[date_str] = {
                    'total_cases': int(date_data['New Cases'].sum()),
                    'max_province': date_data.loc[date_data['New Cases'].idxmax(), 'Location'] if not date_data['New Cases'].isna().all() else "Unknown",
                    'max_cases': int(date_data['New Cases'].max()) if not date_data['New Cases'].isna().all() else 0
                }
        
        return df, date_strings, date_metrics
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, [], {}

@st.cache_data(ttl=3600)
def load_shapefile(shapefile_dir):
    """Load and preprocess the shapefile"""
    try:
        # Try different possible shapefile paths
        shapefile_options = [
            f"{shapefile_dir}/IDN_Indonesia_1.shp",
            f"{shapefile_dir}/IDN_adm1.shp",
            f"{shapefile_dir}/indonesia.shp",
            f"{shapefile_dir}/IDN.shp"
        ]
        
        shapefile_path = None
        for option in shapefile_options:
            if os.path.exists(option):
                shapefile_path = option
                break
        
        if shapefile_path is None:
            st.error("Could not find any suitable shapefile")
            return None
            
        # Load shapefile directly
        gdf = gpd.read_file(shapefile_path)
        
        # Store shapefile columns in session state for debugging
        st.session_state['shapefile_columns'] = list(gdf.columns)
        
        # Optional: simplify geometries for faster rendering
        if not gdf.empty:
            gdf = gdf.copy()  # Make a copy to avoid SettingWithCopyWarning
            
        return gdf
    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        st.exception(e)
        return None

# Function to get data for a specific date (very fast)
def get_date_data(raw_df, date_str):
    """Get data for specific date with minimal processing"""
    # Convert string to datetime if needed
    if isinstance(date_str, str):
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            # Try alternative format
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
    else:
        date_obj = date_str
        
    date_data = raw_df[raw_df['Date'] == date_obj].copy()
    return date_data

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
        
    covid_df, date_strings, date_metrics = load_data(CSV_PATH)
    
with st.spinner("Loading map data..."):
    gdf = load_shapefile(SHAPEFILE_DIR)

# Stop if data loading failed
if covid_df is None or gdf is None:
    st.error("Failed to load required data. Please check your data files.")
    
    # Show more helpful error details
    if covid_df is None:
        st.info(f"The CSV file was found but could not be processed. Make sure it has the correct columns (Date, Location, New Cases) and separator (;).")
    
    if gdf is None:
        st.info(f"Could not load shapefile data. Make sure a shapefile exists in the {SHAPEFILE_DIR} directory.")
        
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

# Create two columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Date Selection")
    
    # Date selector
    selected_date_str = st.select_slider(
        "Select date:",
        options=date_strings,
        value=date_strings[-1] if date_strings else None
    )
    
    # Display metrics for selected date
    if selected_date_str in date_metrics:
        metrics = date_metrics[selected_date_str]
        st.metric("Total New Cases", f"{metrics['total_cases']:,}")
        st.metric("Highest Province", metrics['max_province'])
        st.metric("Cases in Highest", f"{metrics['max_cases']:,}")
    
    # Province name column selection
    province_cols = [col for col in gdf.columns if any(name in col.lower() for name in ['name', 'province', 'admin'])]
    if province_cols:
        default_col = next((col for col in ['NAME_1', 'NAME', 'PROVINCE'] if col in province_cols), province_cols[0])
        province_column = st.selectbox(
            "Select province name column:",
            options=province_cols,
            index=province_cols.index(default_col) if default_col in province_cols else 0
        )
    else:
        province_column = 'NAME_1'  # Default if no good columns found
        st.warning(f"No obvious province column found in shapefile. Using {province_column}")
    
    # Add options
    show_table = st.checkbox("Show data table", value=False)

# Function to create map for selected date
def create_covid_map(gdf, date_data, date_str, province_column):
    """Create choropleth map for the selected date"""
    try:
        # Create base map
        m = folium.Map(
            location=[-2.5, 118],  # Indonesia center
            zoom_start=5,
            tiles="CartoDB positron",
            prefer_canvas=True  # For better performance
        )
        
        # Add choropleth - direct approach without conversion
        choropleth = folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            name="choropleth",
            data=date_data,
            columns=['Location', 'New Cases'],
            key_on=f'feature.properties.{province_column}',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"New COVID-19 Cases ({date_str})",
            highlight=True
        ).add_to(m)
        
        # Add tooltips
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
                fields=[province_column],
                aliases=['Province:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
            )
        ).add_to(m)
        
        # Add a title to the map
        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50px; width: 250px; 
        background-color: rgba(255,255,255,0.8); border-radius: 5px; padding: 5px; 
        font-size: 14px; font-weight: bold; text-align: center;">
        COVID-19 New Cases: {date_str}</div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    except Exception as e:
        st.error(f"Error in map creation: {e}")
        return None

with col2:
    # Create and display map
    st.subheader(f"COVID-19 Map for {selected_date_str}")
    
    try:
        # Get data for this date (fast)
        date_data = get_date_data(covid_df, selected_date_str)
        
        # Only create map if there's data
        if not date_data.empty:
            with st.spinner(f"Creating map for {selected_date_str}..."):
                m = create_covid_map(gdf, date_data, selected_date_str, province_column)
                if m:
                    folium_static(m, width=700, height=500)
                else:
                    st.error("Could not generate map.")
        else:
            st.warning(f"No data available for {selected_date_str}")
    except Exception as e:
        st.error(f"Error displaying map: {e}")
        st.exception(e)

# Display data table conditionally
if show_table and covid_df is not None:
    st.subheader("Data Table")
    
    # Get data for selected date
    table_data = get_date_data(covid_df, selected_date_str)[['Location', 'New Cases']].copy()
    table_data = table_data.sort_values('New Cases', ascending=False)
    
    # Display with formatting
    st.dataframe(
        table_data,
        column_config={
            "Location": st.column_config.TextColumn("Province"),
            "New Cases": st.column_config.NumberColumn("New Cases", format="%d")
        },
        use_container_width=True,
        hide_index=True
    )

# Add trend analysis section
st.subheader("COVID-19 Trend Analysis")

# Get top provinces by total cases
@st.cache_data
def get_top_provinces(df, n=5):
    return df.groupby('Location')['New Cases'].sum().nlargest(n).index.tolist()

top_provinces = get_top_provinces(covid_df)

# Province selector
provinces = st.multiselect(
    "Select provinces to compare:",
    options=sorted(covid_df['Location'].unique()),
    default=top_provinces[:3] if top_provinces else []
)

if provinces:
    # Filter data for selected provinces and create trend plot
    trend_data = []
    
    for province in provinces:
        province_data = covid_df[covid_df['Location'] == province].copy()
        
        # Downsample if too many points (for performance)
        if len(province_data) > 50:
            province_data = province_data.iloc[::len(province_data)//50 + 1]
            
        trend_data.append(province_data)
    
    # Combine all province data
    if trend_data:
        trend_df = pd.concat(trend_data)
        
        # Create plot with Plotly
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

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #666;">
    <p>Data source: Indonesia Ministry of Health • Last updated: May 2025</p>
</div>
""", unsafe_allow_html=True)

# Add debug and performance section (hidden by default)
with st.expander("Debug & Performance Information", expanded=False):
    st.subheader("Performance Metrics")
    st.write("Total number of dates:", len(date_strings))
    st.write("Total number of provinces:", covid_df['Location'].nunique())
    st.write("Total data points:", len(covid_df))
    
    # Show shapefile columns
    if 'shapefile_columns' in st.session_state:
        st.write("Shapefile Columns:")
        st.write(st.session_state['shapefile_columns'])
        
    # Helper function to safely display GeoDataFrame (without geometry issues)
    def safe_display(df):
        df_copy = df.copy()
        if 'geometry' in df_copy.columns:
            df_copy['geometry'] = df_copy['geometry'].astype(str)
        return df_copy
    
    # Show sample of shapefile data
    st.write("Sample of shapefile data:")
    if gdf is not None:
        st.write(safe_display(gdf.head()))
