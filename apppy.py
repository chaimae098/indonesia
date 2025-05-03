import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from datetime import datetime
import plotly.express as px
import os
import numpy as np

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
with open("style.css", "w") as f:
    f.write("""
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
    """)
    
st.markdown(f'<style>{open("style.css").read()}</style>', unsafe_allow_html=True)

# Title
st.title("Indonesia COVID-19 New Cases Choropleth Map")

# --- OPTIMIZED DATA LOADING ---

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(csv_path):
    """Load and preprocess COVID data with optimal performance"""
    try:
        # Specify dtypes for faster loading and less memory usage
        dtypes = {
            'Location': 'category',  # Use category for strings that repeat
            'New Cases': 'float32'   # Use smaller numeric type
        }
        parse_dates = ['Date']
        
        # Load CSV with optimized parameters
        df = pd.read_csv(
            csv_path, 
            sep=';', 
            dtype=dtypes,
            parse_dates=parse_dates,
            low_memory=True
        )
        
        # Drop national data and NaN cases in one operation
        df = df[
            (df['Location'] != 'Indonesia') & 
            (~df['New Cases'].isna())
        ]
        
        # Extract unique dates more efficiently
        all_dates = pd.DatetimeIndex(df['Date'].unique()).sort_values()
        date_strings = all_dates.strftime('%Y-%m-%d').tolist()
        
        # Pre-compute date metrics efficiently by date
        date_data = df.groupby(['Date'])
        
        # Calculate metrics for each date in one go
        max_cases_idx = date_data['New Cases'].idxmax()
        sum_cases = date_data['New Cases'].sum()
        
        # Create metrics dictionary efficiently
        date_metrics = {}
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            date_df = df[df['Date'] == date]
            
            if not date_df.empty:
                max_idx = date_df['New Cases'].idxmax() if len(date_df) > 0 else None
                
                date_metrics[date_str] = {
                    'total_cases': int(date_df['New Cases'].sum()),
                    'max_province': date_df.loc[max_idx, 'Location'] if max_idx is not None else "Unknown",
                    'max_cases': int(date_df['New Cases'].max()) if max_idx is not None else 0
                }
        
        return df, date_strings, date_metrics
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, [], {}

@st.cache_data(ttl=3600, show_spinner=False)
def load_shapefile(shapefile_dir, simplify_tolerance=0.001):
    """Load and preprocess the shapefile with simplification for faster rendering"""
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
            return None, None, []
            
        # Load shapefile with optimized parameters
        gdf = gpd.read_file(shapefile_path, driver='ESRI Shapefile')
        
        # Simplify geometries for MUCH faster rendering
        gdf['geometry'] = gdf['geometry'].simplify(simplify_tolerance)
        
        # Store shapefile columns
        shapefile_columns = list(gdf.columns)
        
        # Convert to GeoJSON string representation (hashable) instead of dict
        gdf_json = gdf.to_json()
        
        return gdf, gdf_json, shapefile_columns
    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        return None, None, []

@st.cache_data(show_spinner=False)
def get_date_data(date_str, data_df):
    """Get data for specific date with minimal processing"""
    # Convert string to datetime for filtering
    if isinstance(date_str, str):
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            # Try alternative format
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
    else:
        date_obj = date_str
    
    # Filter the data efficiently
    mask = data_df['Date'] == date_obj
    date_data = data_df.loc[mask].copy()
    return date_data

@st.cache_data(show_spinner=False)
def get_top_provinces(df, n=5):
    """Calculate top provinces by total cases"""
    return df.groupby('Location')['New Cases'].sum().nlargest(n).index.tolist()

# Check if data files exist
if not os.path.exists(CSV_PATH):
    st.error(f"Data file not found: {CSV_PATH}")
    st.info("Please make sure your data is in the correct location. The app expects data in: data/indonesia_data.csv")
    
    # Create sample data option
    create_sample = st.button("Create sample data file for demonstration")
    
    if create_sample:
        st.info("Creating sample data file...")
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        
        # Generate sample data efficiently
        base_date = datetime(2023, 1, 1)
        provinces = ['Java', 'Sumatra', 'Sulawesi', 'Kalimantan', 'Papua']
        
        # Pre-allocate data arrays for better performance
        dates_arr = []
        loc_arr = []
        cases_arr = []
        
        for i in range(30):
            for province in provinces:
                dates_arr.append((base_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d'))
                loc_arr.append(province)
                cases_arr.append(int(np.random.randint(0, 100)))
        
        # Create DataFrame from arrays (faster than appending rows)
        sample_df = pd.DataFrame({
            'Date': dates_arr,
            'Location': loc_arr,
            'New Cases': cases_arr
        })
        
        # Save to CSV
        sample_df.to_csv(CSV_PATH, sep=';', index=False)
        st.success(f"Sample data created at {CSV_PATH}")
        st.info("Refresh the page to use the sample data")
    
    st.stop()

# Load data using progress indicator
with st.spinner("Loading data..."):
    # Use a combined progress bar for better UX
    progress_bar = st.progress(0)
    
    # Load the data
    progress_bar.progress(25)
    covid_df, date_strings, date_metrics = load_data(CSV_PATH)
    
    progress_bar.progress(50)
    gdf, gdf_json, shapefile_columns = load_shapefile(SHAPEFILE_DIR)
    
    progress_bar.progress(100)
    progress_bar.empty()  # Remove progress bar when done

# Stop if data loading failed
if covid_df is None or gdf is None:
    st.error("Failed to load required data. Please check your data files.")
    
    if covid_df is None:
        st.info(f"The CSV file was found but could not be processed. Make sure it has the correct columns (Date, Location, New Cases) and separator (;).")
    
    if gdf is None:
        st.info(f"Could not load shapefile data. Make sure a shapefile exists in the {SHAPEFILE_DIR} directory.")
        
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

# Create layout with columns
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Date Selection")
    
    # Efficient date selector
    selected_date_str = st.select_slider(
        "Select date:",
        options=date_strings,
        value=date_strings[-1] if date_strings else None
    )
    
    # Display metrics for selected date efficiently
    if selected_date_str in date_metrics:
        metrics = date_metrics[selected_date_str]
        
        # Use a more efficient layout for metrics
        cols = st.columns(3)
        cols[0].metric("Total New Cases", f"{metrics['total_cases']:,}")
        cols[1].metric("Highest Province", metrics['max_province'])
        cols[2].metric("Cases in Highest", f"{metrics['max_cases']:,}")
    
    # Province name column selection
    province_cols = [col for col in shapefile_columns if any(name in col.lower() for name in ['name', 'province', 'admin'])]
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
    
    # Add options - use checkbox with default state
    show_table = st.checkbox("Show data table", value=False)

# Function to create map with caching - FIXED VERSION
@st.cache_data(show_spinner=False)
def create_covid_map(_gdf_data, date_data_df, date_str, province_col):
    """Create optimized choropleth map for the selected date
    Note: _gdf_data has underscore prefix to prevent hashing of this parameter
    """
    try:
        # Unpack the data - but keep as separate variables (hashable)
        gdf, gdf_json_str = _gdf_data
        
        # Create base map with optimized parameters
        m = folium.Map(
            location=[-2.5, 118],  # Indonesia center
            zoom_start=5,
            tiles="CartoDB positron",
            prefer_canvas=True,    # For better performance
            disable_3d=True,       # Disable 3D rendering
            zoom_control=False     # Simplify controls
        )
        
        # Convert GeoJSON string back to dict for folium
        gdf_json_dict = folium.GeoJson(gdf_json_str).data
        
        # Add choropleth with optimized rendering
        choropleth = folium.Choropleth(
            geo_data=gdf_json_dict,
            name="choropleth",
            data=date_data_df,
            columns=['Location', 'New Cases'],
            key_on=f'feature.properties.{province_col}',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"New COVID-19 Cases ({date_str})",
            highlight=True,
            smooth_factor=0.5   # Add smoothing for performance
        ).add_to(m)
        
        # Add simplified tooltips
        style_function = lambda x: {'fillColor': '#ffffff', 
                                   'color':'#000000', 
                                   'fillOpacity': 0.1, 
                                   'weight': 0.1}
        highlight_function = lambda x: {'fillColor': '#000000', 
                                       'color':'#000000', 
                                       'fillOpacity': 0.50, 
                                       'weight': 0.1}
        
        # Use the GeoJSON dict for tooltips
        folium.GeoJson(
            gdf_json_dict,
            style_function=style_function,
            control=False,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=[province_col],
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
        st.error(f"Error in map creation: {str(e)}")
        return None

with col2:
    # Create and display map
    st.subheader(f"COVID-19 Map for {selected_date_str}")
    
    # Get data for this date (fast with caching)
    date_data = get_date_data(selected_date_str, covid_df)
    
    # Only create map if there's data (and hide spinner for better UX)
    if not date_data.empty:
        try:
            # Disable caching temporarily if we're still having issues
            # Create map directly without caching
            try:
                # Create base map with optimized parameters
                m = folium.Map(
                    location=[-2.5, 118],  # Indonesia center
                    zoom_start=5,
                    tiles="CartoDB positron",
                    prefer_canvas=True
                )
                
                # Create GeoJSON from GDF for folium
                if isinstance(gdf_json, str):
                    # If it's a string, convert to dict
                    gdf_json_dict = folium.GeoJson(gdf_json).data
                else:
                    # If it's already a dict-like object, use directly
                    gdf_json_dict = gdf._geo_interface_
                
                # Add choropleth with optimized rendering
                choropleth = folium.Choropleth(
                    geo_data=gdf_json_dict,
                    name="choropleth",
                    data=date_data,
                    columns=['Location', 'New Cases'],
                    key_on=f'feature.properties.{province_column}',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=f"New COVID-19 Cases ({selected_date_str})",
                    highlight=True,
                    smooth_factor=0.5
                ).add_to(m)
                
                # Add tooltips
                folium.GeoJson(
                    gdf_json_dict,
                    style_function=lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1},
                    control=False,
                    highlight_function=lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1},
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=[province_column],
                        aliases=['Province:'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                ).add_to(m)
                
                # Add a title
                title_html = f"""
                <div style="position: fixed; top: 10px; left: 50px; width: 250px; 
                background-color: rgba(255,255,255,0.8); border-radius: 5px; padding: 5px; 
                font-size: 14px; font-weight: bold; text-align: center;">
                COVID-19 New Cases: {selected_date_str}</div>
                """
                m.get_root().html.add_child(folium.Element(title_html))
                
                # Render map
                folium_static(m, width=700, height=500)
            except Exception as e1:
                st.error(f"Primary method failed: {str(e1)}")
                st.info("Trying simplified rendering method...")
                
                # Create a very simple map as fallback
                m = folium.Map(
                    location=[-2.5, 118],
                    zoom_start=5
                )
                
                # Add a simple choropleth with minimal options
                # Try to convert GDF directly to avoid string conversion issues
                try:
                    simple_geojson = gdf._geo_interface_
                    
                    folium.Choropleth(
                        geo_data=simple_geojson,
                        data=date_data,
                        columns=['Location', 'New Cases'],
                        key_on=f'feature.properties.{province_column}',
                        fill_color='YlOrRd'
                    ).add_to(m)
                    
                    folium_static(m, width=700, height=500)
                except Exception as e2:
                    st.error(f"Simplified method also failed: {str(e2)}")
                    st.info("Displaying province data table only")
                    
                    # Show data table as last resort
                    st.dataframe(
                        date_data[['Location', 'New Cases']].sort_values('New Cases', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
        except Exception as e:
            st.error(f"Error rendering map: {str(e)}")
            st.info("Please check your data and shapefile formats")
    else:
        st.warning(f"No data available for {selected_date_str}")

# Display data table conditionally
if show_table and covid_df is not None:
    st.subheader("Data Table")
    
    # Get data for selected date - reuse cached data
    table_data = date_data[['Location', 'New Cases']].copy()
    
    # Sort efficiently
    table_data = table_data.sort_values('New Cases', ascending=False)
    
    # Display with optimized formatting
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

# Get top provinces using cached function
top_provinces = get_top_provinces(covid_df)

# Province selector
provinces = st.multiselect(
    "Select provinces to compare:",
    options=sorted(covid_df['Location'].unique()),
    default=top_provinces[:3] if top_provinces else []
)

# Only create plot if provinces are selected (improves performance)
if provinces:
    # Filter data efficiently
    trend_df = covid_df[covid_df['Location'].isin(provinces)].copy()
    
    try:
        # Downsample the data for better performance - safer approach
        if len(trend_df) > 500:  # Only downsample if needed
            # Use a safer resample approach - first reset index
            trend_list = []
            
            # Process each province separately to avoid dimension mismatch
            for province in provinces:
                province_df = trend_df[trend_df['Location'] == province].copy()
                
                # Only process if we have data
                if not province_df.empty:
                    # Set Date as index for resampling
                    province_df = province_df.set_index('Date')
                    
                    # Resample with 3-day frequency and take the mean
                    resampled = province_df.resample('3D')['New Cases'].mean().reset_index()
                    
                    # Add location back
                    resampled['Location'] = province
                    
                    # Add to our list
                    trend_list.append(resampled)
            
            # Combine all provinces back together
            if trend_list:
                trend_df = pd.concat(trend_list)
        
        # Create plot with Plotly - optimize for performance
        fig = px.line(
            trend_df, 
            x='Date', 
            y='New Cases', 
            color='Location',
            title="COVID-19 New Cases Trend",
            labels={"Date": "Date", "New Cases": "New Cases", "Location": "Province"}
        )
    except Exception as e:
        st.error(f"Error processing trend data: {str(e)}")
        # Create a simple plot with the original data as fallback
        try:
            fig = px.line(
                trend_df, 
                x='Date', 
                y='New Cases', 
                color='Location',
                title="COVID-19 New Cases Trend (Raw Data)",
                labels={"Date": "Date", "New Cases": "New Cases", "Location": "Province"}
            )
        except Exception as e2:
            st.error(f"Could not create trend chart: {str(e2)}")
            fig = None
    
    # Optimize plot layout
    fig.update_layout(
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='closest',  # Improve hover performance
        # Reduce the number of date ticks for performance
        xaxis=dict(
            tickmode='auto',
            nticks=10
        )
    )
    
    # Render chart
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
    st.write("Shapefile Columns:")
    st.write(shapefile_columns)
    
    # Add error handling debugging section
    st.subheader("Error Handling")
    st.write("1. If the map doesn't load, check:")
    st.write("   - The shapefile has a column named similar to NAME_1 or PROVINCE")
    st.write("   - The province names in your CSV match those in the shapefile")
    st.write("2. If trend analysis doesn't display:")
    st.write("   - Ensure you have data for multiple dates for each province")
    
    # Render a sample of the data for debugging
    st.subheader("Data Sample")
    if covid_df is not None:
        st.write(covid_df.head())
