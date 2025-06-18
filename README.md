# indonesia
## Indonesia COVID-19 Interactive Dashboard

This repository contains a Streamlit application that visualizes COVID-19 new cases across Indonesian provinces using an interactive choropleth map and trend analysis charts. The dashboard leverages optimized data loading, caching, and geospatial rendering techniques for high performance and a responsive user experience.

---

### 📁 Repository Structure

```
├── data/
│   ├── indonesia_data.csv       # COVID-19 case data (semicolon-separated)
│   ├── IDN_Indonesia_1.shp      # Example shapefile(s) for provincial boundaries
│   └── ...                      # Other shapefile variants (IDN_adm1.shp, indonesia.shp, etc.)
├── style.css                    # Custom CSS for styling the Streamlit app
├── app.py                       # Main Streamlit application script
└── README.md                    # Project documentation (this file)
```

---

### ⚙️ Features

* **Interactive Choropleth Map**

  * Displays new COVID-19 cases per province for a selected date
  * Hover tooltips and click popups show province-level details
  * Custom HTML/CSS styling for map titles and instructional overlays

* **Trend Analysis**

  * Line charts comparing new case trends for user-selected provinces
  * Automatic downsampling and performance optimizations for large datasets

* **Optimized Data Handling**

  * `@st.cache_data` decorators with TTL for data and shapefile loading
  * Specified `dtype` and `parse_dates` in `pandas.read_csv` to minimize memory footprint
  * Geometry simplification in GeoPandas for faster rendering

* **User Controls**

  * Date slider to navigate through available dates
  * Province column selector for different shapefile schemas
  * Toggleable data table displaying raw numbers
  * Multi-select dropdown for comparing trending provinces

* **Debug & Performance Panel**

  * Expandable section showing dataset sample, performance metrics, and troubleshooting tips

---

### 🚀 Getting Started

#### Prerequisites

* Python 3.8+
* [Streamlit](https://streamlit.io/)
* [Pandas](https://pandas.pydata.org/)
* [GeoPandas](https://geopandas.org/)
* [Folium](https://python-visualization.github.io/folium/)
* [streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
* [Plotly](https://plotly.com/python/)

Install dependencies via:

```bash
pip install streamlit pandas geopandas folium streamlit-folium plotly
```

#### Data Preparation

1. Place your COVID-19 data CSV in `data/indonesia_data.csv` with columns:

   * `Date` (YYYY-MM-DD)
   * `Location` (Province name)
   * `New Cases` (numeric)

2. Add one of the following shapefiles to the `data/` directory:

   * `IDN_Indonesia_1.shp`
   * `IDN_adm1.shp`
   * `indonesia.shp`
   * `IDN.shp`

Ensure that the province names in your CSV match those in the shapefile.

#### Running the App

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

### 🛠️ Configuration

* **`CSV_PATH`**: Path to your COVID-19 data file (default: `data/indonesia_data.csv`).
* **`SHAPEFILE_DIR`**: Directory containing shapefiles (default: `data`).
* **Caching TTL**: Data and shapefiles are cached for one hour by default.
* **Geometry Simplification**: Adjust `simplify_tolerance` in `load_shapefile` for map performance vs. detail.

Customize these settings in `app.py` as needed.

---

### 💡 Usage Tips

* Use the date slider on the sidebar to explore daily case distributions.
* Toggle the raw data table for precise values.
* Select multiple provinces in the trend analysis section to compare case trajectories.
* Expand the Debug & Performance panel to inspect underlying data and troubleshoot issues.

---

### 📜 License & Credits

Developed by **Kazoury Chaimae**.

Data source: Indonesia Ministry of Health (Last updated: May 2025).

---

*Thank you for using the Indonesia COVID-19 Interactive Dashboard!*
