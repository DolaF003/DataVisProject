import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Global Sustainable Energy Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2E8B57, #228B22, #32CD32);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #2E8B57;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .section-header {
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 3rem;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 3px solid #2E8B57;
        padding-bottom: 1rem;
    }
    .big-metric {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0.5rem;
    }
    .stSelectbox > div > div > select {
        font-size: 1.1rem;
    }
    .chart-container {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('global-data-on-sustainable-energy (1).csv')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle the density column name issue
    if 'Density\n(P/Km2)' in df.columns:
        df = df.rename(columns={'Density\n(P/Km2)': 'Population Density (P/Km2)'})
    
    # Convert to proper data types
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    
    return df

# Load data
df = load_data()

# Main title with enhanced styling
st.markdown('<h1 class="main-header">🌍 Global Sustainable Energy Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 3rem;">Explore global energy transitions, sustainability trends, and climate impact across 176 countries</p>', unsafe_allow_html=True)

# Enhanced sidebar with better styling
st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Year range selector with better styling
st.sidebar.markdown("### 📅 Time Period")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=2000,
    max_value=2020,
    value=(2010, 2020),
    step=1,
    help="Choose the time period for analysis"
)

st.sidebar.markdown("### 🌎 Geographic Selection")
# Enhanced country selector
countries = sorted(df['Entity'].unique())
default_countries = ['United States', 'China', 'Germany', 'India', 'Brazil', 'Japan', 'United Kingdom']
selected_countries = st.sidebar.multiselect(
    "Select Countries/Regions",
    countries,
    default=[c for c in default_countries if c in countries],
    help="Choose countries to compare and analyze"
)

st.sidebar.markdown("### 📊 Analysis Focus")
# Enhanced metric selector
metrics = [
    'Access to electricity (% of population)',
    'Renewable energy share in the total final energy consumption (%)',
    'Value_co2_emissions_kt_by_country',
    'gdp_per_capita',
    'Primary energy consumption per capita (kWh/person)'
]

metric_labels = {
    'Access to electricity (% of population)': '⚡ Electricity Access',
    'Renewable energy share in the total final energy consumption (%)': '🌱 Renewable Energy Share',
    'Value_co2_emissions_kt_by_country': '💨 CO₂ Emissions',
    'gdp_per_capita': '💰 GDP per Capita',
    'Primary energy consumption per capita (kWh/person)': '🔋 Energy Consumption'
}

selected_metric = st.sidebar.selectbox(
    "Primary Metric for Analysis",
    metrics,
    index=1,
    format_func=lambda x: metric_labels.get(x, x)
)

# Filter data based on selections
mask = (df['Year'].dt.year >= year_range[0]) & (df['Year'].dt.year <= year_range[1])
filtered_df = df[mask].copy()

# Enhanced main layout - Key Metrics Section
st.markdown('<div class="section-header">📈 Global Energy Overview</div>', unsafe_allow_html=True)

# Create impressive metrics display
latest_year = filtered_df['Year'].max()
latest_data = filtered_df[filtered_df['Year'] == latest_year]

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_electricity = latest_data['Access to electricity (% of population)'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-metric">{avg_electricity:.1f}%</div>
        <div class="metric-label">Global Electricity Access</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_renewable = latest_data['Renewable energy share in the total final energy consumption (%)'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-metric">{avg_renewable:.1f}%</div>
        <div class="metric-label">Renewable Energy Share</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_co2 = latest_data['Value_co2_emissions_kt_by_country'].sum() / 1000000
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-metric">{total_co2:.1f}B</div>
        <div class="metric-label">Global CO₂ Emissions (kt)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    countries_with_data = latest_data['Entity'].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-metric">{countries_with_data}</div>
        <div class="metric-label">Countries Tracked</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced World Map - MASSIVE AND PROMINENT
st.markdown('<div class="section-header">🗺️ Interactive Global Energy Map</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">🎯 Hover over countries to explore detailed energy data | 🔍 Click and drag to zoom | 📊 Bubble size shows energy consumption</p>', unsafe_allow_html=True)

map_data = latest_data.copy()
map_data = map_data.dropna(subset=[selected_metric, 'Latitude', 'Longitude'])

# Ensure bubble sizes are visible and reasonable
map_data['Bubble_Size'] = map_data['Primary energy consumption per capita (kWh/person)'].fillna(1000)
map_data['Bubble_Size'] = np.where(map_data['Bubble_Size'] < 500, 500, map_data['Bubble_Size'])

# Create stunning world map with enhanced tooltips
fig_map = px.scatter_geo(
    map_data,
    lat='Latitude',
    lon='Longitude',
    color=selected_metric,
    size='Bubble_Size',
    hover_name='Entity',
    hover_data={
        'Access to electricity (% of population)': ':.1f',
        'Renewable energy share in the total final energy consumption (%)': ':.1f',
        'Value_co2_emissions_kt_by_country': ':.0f',
        'gdp_per_capita': ':.0f',
        'Primary energy consumption per capita (kWh/person)': ':.0f',
        'Latitude': False,
        'Longitude': False,
        'Bubble_Size': False
    },
    title=f"<b>Global {metric_labels.get(selected_metric, selected_metric)} - {latest_year.year}</b>",
    color_continuous_scale="Viridis",
    projection="natural earth",
    size_max=60
)

# Simplified map styling to avoid errors
fig_map.update_traces(
    marker=dict(
        opacity=0.8,
        line=dict(width=1, color='white'),
        sizemin=5
    )
)

fig_map.update_layout(
    height=800,  # Large but stable height
    title_font_size=24,
    title_x=0.5,
    geo=dict(
        showframe=False, 
        showcoastlines=True,
        landcolor='lightgray',
        oceancolor='lightblue',
        projection_type='natural earth'
    ),
    font=dict(size=14),
    margin=dict(l=20, r=20, t=80, b=20)
)

# Display the map with simplified config
st.plotly_chart(fig_map, use_container_width=True, key="world_map")

# Enhanced Time Series Analysis - MUCH LARGER
st.markdown('<div class="section-header">📊 Time Series Analysis</div>', unsafe_allow_html=True)

if selected_countries:
    country_data = filtered_df[filtered_df['Entity'].isin(selected_countries)]
    
    # Create two large charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced time series chart
        fig_timeseries = px.line(
            country_data,
            x='Year',
            y=selected_metric,
            color='Entity',
            title=f"<b>{metric_labels.get(selected_metric, selected_metric)} Trends</b>",
            hover_data=['gdp_per_capita', 'Primary energy consumption per capita (kWh/person)'],
            line_shape='spline'
        )
        
        fig_timeseries.update_layout(
            height=600,  # Much larger
            title_font_size=20,
            title_x=0.5,
            xaxis_title="Year",
            yaxis_title=selected_metric,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            font=dict(size=12),
            showlegend=True,
            hovermode='x unified'
        )
        
        fig_timeseries.update_traces(line=dict(width=3))
        st.plotly_chart(fig_timeseries, use_container_width=True, key="timeseries")
    
    with col2:
        # Enhanced correlation analysis
        st.markdown("#### 🔗 Renewable Energy vs Development")
        
        # Create scatter with trend line
        scatter_data = country_data.dropna(subset=['Renewable energy share in the total final energy consumption (%)', 'gdp_per_capita'])
        
        fig_scatter_trend = px.scatter(
            scatter_data,
            x='gdp_per_capita',
            y='Renewable energy share in the total final energy consumption (%)',
            color='Entity',
            size='Primary energy consumption per capita (kWh/person)',
            animation_frame='Year',
            title="<b>Economic Development vs Renewable Energy</b>",
            hover_data=['Access to electricity (% of population)'],
            trendline="ols"
        )
        
        fig_scatter_trend.update_layout(
            height=600,  # Much larger
            title_font_size=20,
            title_x=0.5,
            xaxis_title="GDP per Capita ($)",
            yaxis_title="Renewable Energy Share (%)",
            font=dict(size=12),
            showlegend=True
        )
        
        st.plotly_chart(fig_scatter_trend, use_container_width=True, key="scatter_trend")

# Enhanced Multi-Metric Dashboard - FULL WIDTH
st.markdown('<div class="section-header">⚖️ Comprehensive Multi-Metric Analysis</div>', unsafe_allow_html=True)

if selected_countries:
    # Create comprehensive dashboard
    fig_dashboard = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '⚡ Electricity Access (%)',
            '🌱 Renewable Energy Share (%)',
            '💨 CO₂ Emissions (kt)',
            '💰 GDP per Capita ($)',
            '🔋 Energy Consumption (kWh/person)',
            '🏭 Energy Intensity (MJ/$)'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1[:len(selected_countries)]
    
    for i, country in enumerate(selected_countries):
        country_data = filtered_df[filtered_df['Entity'] == country]
        
        if not country_data.empty:
            # Row 1: Access, Renewable, CO2
            fig_dashboard.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Access to electricity (% of population)'],
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=True,
                    hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Access: %{{y:.1f}}%<extra></extra>"
                ),
                row=1, col=1
            )
            
            fig_dashboard.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Renewable energy share in the total final energy consumption (%)'],
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=False,
                    hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Renewable: %{{y:.1f}}%<extra></extra>"
                ),
                row=1, col=2
            )
            
            fig_dashboard.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Value_co2_emissions_kt_by_country'],
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=False,
                    hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>CO₂: %{{y:.0f}} kt<extra></extra>"
                ),
                row=1, col=3
            )
            
            # Row 2: GDP, Energy Consumption, Energy Intensity
            fig_dashboard.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['gdp_per_capita'],
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=False,
                    hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>GDP: $%{{y:.0f}}<extra></extra>"
                ),
                row=2, col=1
            )
            
            fig_dashboard.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Primary energy consumption per capita (kWh/person)'],
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=False,
                    hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Consumption: %{{y:.0f}} kWh<extra></extra>"
                ),
                row=2, col=2
            )
            
            fig_dashboard.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'],
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=False,
                    hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Intensity: %{{y:.2f}} MJ/$<extra></extra>"
                ),
                row=2, col=3
            )
    
    fig_dashboard.update_layout(
        height=800,  # Much larger
        title_text="<b>Comprehensive Country Energy & Economic Comparison</b>",
        title_font_size=24,
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=12)
    )
    
    # Update subplot titles font size
    fig_dashboard.update_annotations(font_size=16)
    
    st.plotly_chart(fig_dashboard, use_container_width=True, key="multi_metric_dashboard")

# Enhanced Regional Analysis
st.markdown('<div class="section-header">🌐 Regional Performance Analysis</div>', unsafe_allow_html=True)

# Create regional groupings (simplified)
region_mapping = {
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'China': 'Asia', 'India': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia',
    'Germany': 'Europe', 'United Kingdom': 'Europe', 'France': 'Europe', 'Italy': 'Europe',
    'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
    'Australia': 'Oceania', 'New Zealand': 'Oceania'
}

# Add region column to data
latest_data_with_regions = latest_data.copy()
latest_data_with_regions['Region'] = latest_data_with_regions['Entity'].map(region_mapping)
latest_data_with_regions = latest_data_with_regions.dropna(subset=['Region'])

if not latest_data_with_regions.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional comparison bar chart
        regional_avg = latest_data_with_regions.groupby('Region')[selected_metric].mean().reset_index()
        regional_avg = regional_avg.sort_values(selected_metric, ascending=False)
        
        fig_regional = px.bar(
            regional_avg,
            x='Region',
            y=selected_metric,
            title=f"<b>Regional Average: {metric_labels.get(selected_metric, selected_metric)}</b>",
            color=selected_metric,
            color_continuous_scale="Viridis"
        )
        
        fig_regional.update_layout(
            height=500,
            title_font_size=18,
            title_x=0.5,
            xaxis_title="Region",
            yaxis_title=selected_metric,
            font=dict(size=12),
            showlegend=False
        )
        
        st.plotly_chart(fig_regional, use_container_width=True, key="regional_comparison")
    
    with col2:
        # Top performers chart
        top_countries = latest_data.nlargest(10, selected_metric)[['Entity', selected_metric]]
        
        fig_top = px.bar(
            top_countries,
            x=selected_metric,
            y='Entity',
            orientation='h',
            title=f"<b>Top 10 Countries: {metric_labels.get(selected_metric, selected_metric)}</b>",
            color=selected_metric,
            color_continuous_scale="Greens"
        )
        
        fig_top.update_layout(
            height=500,
            title_font_size=18,
            title_x=0.5,
            xaxis_title=selected_metric,
            yaxis_title="Country",
            font=dict(size=12),
            showlegend=False
        )
        
        st.plotly_chart(fig_top, use_container_width=True, key="top_performers")

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; background: linear-gradient(90deg, #f0f2f6, #e8eaf6); padding: 2rem; border-radius: 10px; margin-top: 3rem;'>
    <h3 style='color: #2E8B57; margin-bottom: 1rem;'>🌱 Sustainable Energy Insights Dashboard</h3>
    <p style='font-size: 1.1rem; color: #666; margin-bottom: 1rem;'>
        Interactive exploration of global energy transitions across {len(countries)} countries and regions
    </p>
    <p style='color: #888;'>
        📊 Built with Streamlit & Plotly | 🌍 Data covers 2000-2020 | 
        💡 Hover, click, and interact with all visualizations
    </p>
</div>
""", unsafe_allow_html=True)
