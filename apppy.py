import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from datetime import datetime

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
@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def load_data(csv_path, shapefile_path):
    try:
        # Load CSV - parse dates directly when loading
        df = pd.read_csv(csv_path, sep=';', parse_dates=['Date'])
        df['New Cases'] = pd.to_numeric(df['New Cases'], errors='coerce')
        df = df[df['Location'] != 'Indonesia']  # drop country-wide aggregate
        
        # Create pivot table - keeping original logic
        pivot = df.pivot(index='Date', columns='Location', values='New Cases').reset_index()
        
        # Convert dates to string format
        date_strings = pivot['Date'].dt.strftime('%m/%d/%Y').tolist()
        
        # Load shapefile - more efficient by specifying dtype
        shp = gpd.read_file(shapefile_path)
        
        # Process for map using original logic but with minor optimizations
        merged_gdf = None
        for date, date_str in zip(pivot['Date'], date_strings):
            # Get data for this date
            date_data = pivot.set_index('Date').loc[date].reset_index()
            date_data.columns = ['Province', 'New_Cases']
            
            # Merge with shapefile
            temp_gdf = shp.merge(date_data, left_on='NAME_1', right_on='Province', how='left')
            temp_gdf[date_str] = temp_gdf['New_Cases']
            
            if merged_gdf is None:
                merged_gdf = temp_gdf[['NAME_1', 'geometry', date_str]]
            else:
                merged_gdf[date_str] = temp_gdf[date_str]
        
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
date_objs = [datetime.strptime(d, '%m/%d/%Y') for d in dates]
sorted_indices = sorted(range(len(date_objs)), key=lambda k: date_objs[k])
dates_sorted = [dates[i] for i in sorted_indices]
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
    
    # Calculate COVID metrics for selected date
    if data_clean is not None:
        try:
            # Convert selected_date string to datetime
            date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
            
            # Get data for this date
            date_data = data_clean.set_index('Date').loc[date_obj]
            
            # Calculate metrics
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

# 3. Create map with caching for better performance
@st.cache_data(ttl=3600)
def create_map(gdf, date):
    """Cache map creation to improve performance"""
    # Create map
    m = folium.Map(
        location=[-2.5, 118], 
        zoom_start=5, 
        tiles="CartoDB positron",
        prefer_canvas=True
    )
    
    # Add choropleth
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
    
    # Add tooltips
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
    st.subheader(f"Choropleth for {selected_date}")
    
    try:
        if gdf is not None and selected_date in gdf.columns:
            with st.spinner(f"Creating map for {selected_date}..."):
                # Use cached map creation
                m = create_map(gdf, selected_date)
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

# Cache table data
@st.cache_data(ttl=3600)
def get_table_data(raw_df, date_obj):
    # Filter data for selected date
    filtered_data = raw_df[raw_df['Date'] == date_obj]
    table_data = filtered_data[['Location', 'New Cases']].copy()
    table_data = table_data[table_data['New Cases'].notna()].sort_values('New Cases', ascending=False)
    return table_data

# Display data table conditionally
if show_table and raw_df is not None:
    st.subheader("Data Table")
    try:
        # Convert selected_date string to pandas datetime
        date_obj = datetime.strptime(selected_date, '%m/%d/%Y')
        
        # Get cached table data
        table_data = get_table_data(raw_df, date_obj)
        
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

# Cache trend data
@st.cache_data(ttl=3600)
def get_trend_data(raw_df, provinces):
    if not provinces:
        return None
    return raw_df[raw_df['Location'].isin(provinces)]

# Add trend analysis section
st.subheader("COVID-19 Trend Analysis")
if raw_df is not None:
    # Get default provinces
    @st.cache_data(ttl=3600)
    def get_default_provinces(df):
        return df['Location'].value_counts().nlargest(3).index.tolist()
    
    default_provinces = get_default_provinces(raw_df)
    
    # Create a simple trend chart for selected provinces
    provinces = st.multiselect(
        "Select provinces to compare:",
        options=sorted(raw_df['Location'].unique()),
        default=default_provinces
    )
    
    if provinces:
        # Get cached trend data
        trend_data = get_trend_data(raw_df, provinces)
        
        # Convert to plotly-friendly format
        import plotly.express as px
        
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
