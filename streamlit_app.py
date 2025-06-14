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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .section-header {
        color: #1f4e79;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
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

# Main title
st.markdown('<h1 class="main-header">🌍 Global Sustainable Energy Dashboard</h1>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🎛️ Dashboard Controls")

# Year range selector
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=2000,
    max_value=2020,
    value=(2010, 2020),
    step=1
)

# Country/Region selector
countries = sorted(df['Entity'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries/Regions",
    countries,
    default=['United States', 'China', 'Germany', 'India', 'Brazil']
)

# Metric selector for comparison
metrics = [
    'Access to electricity (% of population)',
    'Renewable energy share in the total final energy consumption (%)',
    'Value_co2_emissions_kt_by_country',
    'gdp_per_capita',
    'Primary energy consumption per capita (kWh/person)'
]

selected_metric = st.sidebar.selectbox(
    "Primary Metric for Analysis",
    metrics,
    index=1
)

# Filter data based on selections
mask = (df['Year'].dt.year >= year_range[0]) & (df['Year'].dt.year <= year_range[1])
filtered_df = df[mask].copy()

# Create main layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">🗺️ Global Overview</div>', unsafe_allow_html=True)
    
    # World map visualization
    latest_year = filtered_df['Year'].max()
    map_data = filtered_df[filtered_df['Year'] == latest_year].copy()
    
    fig_map = px.scatter_geo(
        map_data,
        lat='Latitude',
        lon='Longitude',
        color=selected_metric,
        size='Primary energy consumption per capita (kWh/person)',
        hover_name='Entity',
        hover_data={
            'Access to electricity (% of population)': ':.1f',
            'Renewable energy share in the total final energy consumption (%)': ':.1f',
            'Value_co2_emissions_kt_by_country': ':.0f',
            'Latitude': False,
            'Longitude': False
        },
        title=f"Global {selected_metric} Distribution ({latest_year.year})",
        color_continuous_scale="Viridis",
        projection="natural earth"
    )
    
    fig_map.update_layout(
        height=500,
        geo=dict(showframe=False, showcoastlines=True)
    )
    
    # Add click event handling
    map_selection = st.plotly_chart(fig_map, use_container_width=True, key="world_map")

with col2:
    st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
    
    # Summary statistics
    if not filtered_df.empty:
        latest_data = filtered_df[filtered_df['Year'] == latest_year]
        
        avg_electricity_access = latest_data['Access to electricity (% of population)'].mean()
        avg_renewable_share = latest_data['Renewable energy share in the total final energy consumption (%)'].mean()
        total_co2 = latest_data['Value_co2_emissions_kt_by_country'].sum()
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Global Averages ({latest_year.year})</h3>
            <p><strong>Electricity Access:</strong> {avg_electricity_access:.1f}%</p>
            <p><strong>Renewable Energy Share:</strong> {avg_renewable_share:.1f}%</p>
            <p><strong>Total CO₂ Emissions:</strong> {total_co2/1000:.1f}M kt</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top/Bottom performers
    st.markdown("### 🏆 Top Performers")
    if selected_metric in latest_data.columns:
        top_countries = latest_data.nlargest(5, selected_metric)[['Entity', selected_metric]]
        for idx, row in top_countries.iterrows():
            st.write(f"{row['Entity']}: {row[selected_metric]:.1f}")

# Time series analysis section
st.markdown('<div class="section-header">📈 Time Series Analysis</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    # Multi-country time series
    if selected_countries:
        country_data = filtered_df[filtered_df['Entity'].isin(selected_countries)]
        
        fig_timeseries = px.line(
            country_data,
            x='Year',
            y=selected_metric,
            color='Entity',
            title=f"{selected_metric} Over Time",
            hover_data=['gdp_per_capita', 'Primary energy consumption per capita (kWh/person)']
        )
        
        fig_timeseries.update_layout(
            height=400,
            xaxis_title="Year",
            yaxis_title=selected_metric,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_timeseries, use_container_width=True, key="timeseries")

with col4:
    # Correlation heatmap
    st.subheader("🔗 Metric Correlations")
    
    correlation_metrics = [
        'Access to electricity (% of population)',
        'Renewable energy share in the total final energy consumption (%)',
        'Value_co2_emissions_kt_by_country',
        'gdp_per_capita',
        'Primary energy consumption per capita (kWh/person)'
    ]
    
    corr_data = filtered_df[correlation_metrics].corr()
    
    fig_heatmap = px.imshow(
        corr_data,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        title="Correlation Matrix",
        aspect="auto"
    )
    
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True, key="correlation")

# Comparative analysis section
st.markdown('<div class="section-header">⚖️ Comparative Analysis</div>', unsafe_allow_html=True)

# Interactive scatter plot with brushing
col5, col6 = st.columns([3, 1])

with col5:
    x_metric = st.selectbox(
        "X-axis Metric",
        metrics,
        index=3,
        key="x_axis"
    )
    
    y_metric = st.selectbox(
        "Y-axis Metric",
        metrics,
        index=0,
        key="y_axis"
    )
    
    # Create scatter plot with selection capability
    scatter_data = filtered_df[filtered_df['Year'] == latest_year].copy()
    
    fig_scatter = px.scatter(
        scatter_data,
        x=x_metric,
        y=y_metric,
        color='Renewable energy share in the total final energy consumption (%)',
        size='Primary energy consumption per capita (kWh/person)',
        hover_name='Entity',
        hover_data={
            'Access to clean fuels for cooking': ':.1f',
            'Value_co2_emissions_kt_by_country': ':.0f'
        },
        title=f"{y_metric} vs {x_metric}",
        color_continuous_scale="Viridis"
    )
    
    fig_scatter.update_layout(height=500)
    scatter_selection = st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")

with col6:
    st.subheader("🎯 Selection Tools")
    
    # Region filter
    region_filter = st.multiselect(
        "Filter by Region",
        ['Europe', 'Asia', 'Africa', 'Americas', 'Oceania'],
        default=[],
        help="Select regions to highlight"
    )
    
    # Data export option
    if st.button("📊 Export Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sustainable_energy_data_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )

# Detailed country analysis
with st.expander("🔍 Detailed Country Analysis", expanded=False):
    if selected_countries:
        st.subheader("Comprehensive Country Comparison")
        
        # Create subplot for multiple metrics
        fig_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Electricity Access (%)',
                'Renewable Energy Share (%)',
                'CO₂ Emissions (kt)',
                'GDP per Capita'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1[:len(selected_countries)]
        
        for i, country in enumerate(selected_countries):
            country_data = filtered_df[filtered_df['Entity'] == country]
            
            # Row 1, Col 1: Electricity Access
            fig_multi.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Access to electricity (% of population)'],
                    name=country,
                    line=dict(color=colors[i]),
                    showlegend=True if i == 0 else False
                ),
                row=1, col=1
            )
            
            # Row 1, Col 2: Renewable Energy Share
            fig_multi.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Renewable energy share in the total final energy consumption (%)'],
                    name=country,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Row 2, Col 1: CO2 Emissions
            fig_multi.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Value_co2_emissions_kt_by_country'],
                    name=country,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Row 2, Col 2: GDP per Capita
            fig_multi.add_trace(
                go.Scatter(
                    x=country_data['Year'],
                    y=country_data['gdp_per_capita'],
                    name=country,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig_multi.update_layout(
            height=600,
            title_text="Multi-Metric Country Comparison",
            showlegend=True
        )
        
        st.plotly_chart(fig_multi, use_container_width=True, key="multi_metric")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌱 Dashboard built with Streamlit and Plotly | Data: Global Sustainable Energy Dataset</p>
    <p>Interact with charts by clicking, hovering, and using the sidebar controls</p>
</div>
""", unsafe_allow_html=True)
