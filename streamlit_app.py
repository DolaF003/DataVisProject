import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Global Carbon Emissions Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌍 Global Carbon Emissions Overview")
st.markdown("""
**Exploring the uneven responsibility for climate change across nations**

This dashboard reveals the stark disparities in carbon emissions between developed and developing countries, 
highlighting how some nations contribute disproportionately to the climate crisis while others struggle 
with basic energy access. (Note: Due to missing emissions data for 2020 for all countires, there will be no charts to be shown for the year 
2020 for the Global Carbon Emissions Overview)
""")

@st.cache_data
def load_and_clean_data():
    """Load and clean the sustainable energy dataset"""
    # Load the actual dataset
    df = pd.read_csv('global-data-on-sustainable-energy (1).csv')
    
    # Clean column names - remove extra whitespace and newlines
    df.columns = df.columns.str.strip().str.replace('\n', ' ')
    
    # Remove rows with missing Entity/Year (essential fields)
    df = df.dropna(subset=['Entity', 'Year'])
    
    # Remove rows with excessive missing values (>50%)
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    
    # Convert Year to int to avoid any issues
    df['Year'] = df['Year'].astype(int)
    
    return df

def create_emissions_map(df, year, metric='total'):
    """Create choropleth map of emissions by country"""
    
    # Filter data for selected year and remove NaN values
    year_data = df[df['Year'] == year].copy()
    
    if metric == 'total':
        # Remove rows with missing CO2 data
        year_data = year_data.dropna(subset=['Value_co2_emissions_kt_by_country'])
        color_col = 'Value_co2_emissions_kt_by_country'
        title = f"Total CO₂ Emissions by Country ({year})"
        color_label = "CO₂ Emissions (kt)"
    elif metric == 'per_gdp':
        # Remove rows with missing CO2 or GDP data
        year_data = year_data.dropna(subset=['Value_co2_emissions_kt_by_country', 'gdp_per_capita'])
        # Avoid division by zero
        year_data = year_data[year_data['gdp_per_capita'] > 0]
        year_data['emissions_per_gdp'] = year_data['Value_co2_emissions_kt_by_country'] / year_data['gdp_per_capita']
        color_col = 'emissions_per_gdp'
        title = f"CO₂ Emissions per GDP by Country ({year})"
        color_label = "CO₂ per GDP (kt per capita)"
    else:  # per_capita (simplified - would need population data)
        year_data = year_data.dropna(subset=['Value_co2_emissions_kt_by_country'])
        year_data['emissions_proxy'] = year_data['Value_co2_emissions_kt_by_country'] / 1000  # Simplified
        color_col = 'emissions_proxy'
        title = f"CO₂ Emissions Intensity by Country ({year})"
        color_label = "Emission Intensity (proxy)"
    
    if len(year_data) == 0:
        # Create empty map if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected year and metric",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=500, title=title)
        return fig
    
    # Create choropleth map
    fig = px.choropleth(
        year_data,
        locations='Entity',
        locationmode='country names',
        color=color_col,
        hover_name='Entity',
        hover_data={
            'Value_co2_emissions_kt_by_country': ':,.0f',
            'gdp_per_capita': ':,.0f',
            'Access to electricity (% of population)': ':.1f'
        },
        color_continuous_scale='Reds',
        title=title,
        labels={color_col: color_label}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        height=500,
        title_x=0.5
    )
    
    return fig

def create_top_emitters_chart(df, year, top_n=10):
    """Create bar chart of top emitters"""
    year_data = df[df['Year'] == year].copy()
    
    # Remove NaN values and get top emitters
    year_data = year_data.dropna(subset=['Value_co2_emissions_kt_by_country'])
    
    if len(year_data) == 0:
        # Create empty chart if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected year",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=400, title=f"Top {top_n} CO₂ Emitters ({year})")
        return fig
    
    top_emitters = year_data.nlargest(min(top_n, len(year_data)), 'Value_co2_emissions_kt_by_country')
    
    fig = px.bar(
        top_emitters,
        x='Value_co2_emissions_kt_by_country',
        y='Entity',
        orientation='h',
        title=f"Top {min(top_n, len(year_data))} CO₂ Emitters ({year})",
        labels={'Value_co2_emissions_kt_by_country': 'CO₂ Emissions (kt)', 'Entity': 'Country'},
        color='Value_co2_emissions_kt_by_country',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False
    )
    
    return fig

def create_sdg_map(df, year, indicator):
    """Create map showing SDG-7 progress by country"""
    year_data = df[df['Year'] == year].copy()
    
    if indicator not in df.columns:
        st.error(f"Indicator '{indicator}' not found in dataset.")
        return None
        
    # Remove missing data
    year_data = year_data.dropna(subset=[indicator])
    
    if len(year_data) == 0:
        st.warning(f"No data available for {indicator} in {year}")
        return None
    
    # Define color scale and targets based on indicator
    if indicator == 'Access to electricity (% of population)':
        title = f"Electricity Access Progress ({year})"
        color_label = "% Population with Electricity Access"
        # Reverse color scale - red for low access, green for high
        color_scale = [[0, '#8B0000'], [0.5, '#FF4500'], [0.8, '#FFD700'], [1, '#228B22']]
    elif indicator == 'Access to clean fuels for cooking':
        title = f"Clean Cooking Access Progress ({year})"
        color_label = "% Population with Clean Cooking Access"
        color_scale = [[0, '#8B0000'], [0.5, '#FF4500'], [0.8, '#FFD700'], [1, '#228B22']]
    else:  # renewable share
        title = f"Renewable Energy Share Progress ({year})"
        color_label = "% Renewable Energy Share"
        color_scale = [[0, '#8B0000'], [0.3, '#FF4500'], [0.6, '#FFD700'], [1, '#228B22']]
    
    try:
        fig = px.choropleth(
            year_data,
            locations='Entity',
            locationmode='country names',
            color=indicator,
            hover_name='Entity',
            hover_data={indicator: ':.1f'},
            color_continuous_scale=color_scale,
            title=title,
            labels={indicator: color_label},
            range_color=[0, 100] if 'Access to' in indicator else [0, year_data[indicator].max()]
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=500,
            title_x=0.5
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating SDG map: {str(e)}")
        return None

def create_sdg_gap_analysis(df, year):
    """Create chart showing countries furthest from SDG-7 targets"""
    year_data = df[df['Year'] == year].copy()
    
    # Calculate gaps from targets (100% for access indicators)
    electricity_gap = 100 - year_data['Access to electricity (% of population)'].fillna(0)
    cooking_gap = 100 - year_data['Access to clean fuels for cooking'].fillna(0)
    
    # Create gap analysis
    gap_data = pd.DataFrame({
        'Entity': year_data['Entity'],
        'Electricity_Gap': electricity_gap,
        'Cooking_Gap': cooking_gap,
        'Total_Gap': electricity_gap + cooking_gap
    })
    
    # Get top 15 countries with largest gaps
    top_gaps = gap_data.nlargest(15, 'Total_Gap')
    
    if len(top_gaps) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Electricity Access Gap',
        x=top_gaps['Entity'],
        y=top_gaps['Electricity_Gap'],
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Clean Cooking Gap',
        x=top_gaps['Entity'],
        y=top_gaps['Cooking_Gap'],
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        title=f"Countries with Largest SDG-7 Access Gaps ({year})",
        xaxis_title="Country",
        yaxis_title="Gap to Universal Access (%)",
        barmode='stack',
        height=400,
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_sdg_progress_chart(df, countries, indicator):
    """Create line chart showing SDG progress over time"""
    if not countries:
        return None
        
    country_data = df[df['Entity'].isin(countries)].copy()
    country_data = country_data.dropna(subset=[indicator])
    
    if len(country_data) == 0:
        return None
    
    fig = px.line(
        country_data,
        x='Year',
        y=indicator,
        color='Entity',
        title=f"SDG-7 Progress Over Time: {indicator}",
        labels={indicator: indicator}
    )
    
    # Add target line for access indicators
    if 'Access to' in indicator:
        fig.add_hline(y=100, line_dash="dash", line_color="green", 
                     annotation_text="SDG Target: 100%")
    
    fig.update_layout(height=400)
    return fig

def create_regional_comparison(df, year):
    """Create regional comparison of SDG-7 indicators"""
    year_data = df[df['Year'] == year].copy()
    
    # Define region mapping (expanded)
    region_mapping = {
        'Nigeria': 'Sub-Saharan Africa',
        'Ethiopia': 'Sub-Saharan Africa', 
        'Kenya': 'Sub-Saharan Africa',
        'Chad': 'Sub-Saharan Africa',
        'Madagascar': 'Sub-Saharan Africa',
        'Niger': 'Sub-Saharan Africa',
        'Mali': 'Sub-Saharan Africa',
        'Burkina Faso': 'Sub-Saharan Africa',
        'Mozambique': 'Sub-Saharan Africa',
        'Tanzania': 'Sub-Saharan Africa',
        'Uganda': 'Sub-Saharan Africa',
        'Rwanda': 'Sub-Saharan Africa',
        'Ghana': 'Sub-Saharan Africa',
        'Senegal': 'Sub-Saharan Africa',
        'India': 'South Asia',
        'Bangladesh': 'South Asia',
        'Pakistan': 'South Asia',
        'Afghanistan': 'South Asia',
        'Nepal': 'South Asia',
        'Sri Lanka': 'South Asia',
        'Myanmar': 'South Asia',
        'China': 'East Asia',
        'Mongolia': 'East Asia',
        'North Korea': 'East Asia',
        'Indonesia': 'Southeast Asia',
        'Philippines': 'Southeast Asia',
        'Vietnam': 'Southeast Asia',
        'Cambodia': 'Southeast Asia',
        'Laos': 'Southeast Asia',
        'Thailand': 'Southeast Asia',
        'Malaysia': 'Southeast Asia',
        'Brazil': 'Latin America',
        'Peru': 'Latin America',
        'Bolivia': 'Latin America',
        'Guatemala': 'Latin America',
        'Haiti': 'Latin America',
        'Honduras': 'Latin America',
        'Nicaragua': 'Latin America',
        'Paraguay': 'Latin America',
        'Ecuador': 'Latin America',
        'United States': 'North America',
        'Canada': 'North America',
        'Mexico': 'North America',
        'Germany': 'Europe',
        'France': 'Europe',
        'United Kingdom': 'Europe',
        'Italy': 'Europe',
        'Spain': 'Europe',
        'Poland': 'Europe',
        'Romania': 'Europe',
        'Netherlands': 'Europe',
        'Belgium': 'Europe',
        'Czech Republic': 'Europe',
        'Portugal': 'Europe',
        'Hungary': 'Europe',
        'Sweden': 'Europe',
        'Austria': 'Europe',
        'Belarus': 'Europe',
        'Switzerland': 'Europe',
        'Bulgaria': 'Europe',
        'Serbia': 'Europe',
        'Denmark': 'Europe',
        'Finland': 'Europe',
        'Slovakia': 'Europe',
        'Norway': 'Europe',
        'Ireland': 'Europe',
        'Croatia': 'Europe',
        'Bosnia and Herzegovina': 'Europe',
        'Albania': 'Europe',
        'Lithuania': 'Europe',
        'Slovenia': 'Europe',
        'Latvia': 'Europe',
        'Estonia': 'Europe',
        'Macedonia': 'Europe',
        'Moldova': 'Europe',
        'Russia': 'Europe',
        'Ukraine': 'Europe',
        'Egypt': 'Middle East & North Africa',
        'Iran': 'Middle East & North Africa',
        'Iraq': 'Middle East & North Africa',
        'Saudi Arabia': 'Middle East & North Africa',
        'Turkey': 'Middle East & North Africa',
        'Algeria': 'Middle East & North Africa',
        'Morocco': 'Middle East & North Africa',
        'Tunisia': 'Middle East & North Africa',
        'Jordan': 'Middle East & North Africa',
        'Lebanon': 'Middle East & North Africa',
        'Syria': 'Middle East & North Africa',
        'Yemen': 'Middle East & North Africa',
        'Libya': 'Middle East & North Africa',
        'Australia': 'Oceania',
        'New Zealand': 'Oceania',
        'Papua New Guinea': 'Oceania',
        'Fiji': 'Oceania'
    }
    
    # Add region column
    year_data['Region'] = year_data['Entity'].map(region_mapping).fillna('Other')
    
    # Calculate regional averages
    regional_stats = year_data.groupby('Region').agg({
        'Access to electricity (% of population)': 'mean',
        'Access to clean fuels for cooking': 'mean',
        'Renewable energy share in the total final energy consumption (%)': 'mean'
    }).round(1)
    
    # Filter out regions with no data and Other
    regional_stats = regional_stats.dropna()
    if 'Other' in regional_stats.index:
        regional_stats = regional_stats.drop('Other')
    
    if len(regional_stats) == 0:
        return None, year_data
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Electricity Access (%)',
        x=regional_stats.index,
        y=regional_stats['Access to electricity (% of population)'],
        marker_color='#1f77b4',
        hovertemplate='<b>%{x}</b><br>Electricity Access: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Clean Cooking Access (%)',
        x=regional_stats.index,
        y=regional_stats['Access to clean fuels for cooking'],
        marker_color='#ff7f0e',
        hovertemplate='<b>%{x}</b><br>Clean Cooking Access: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Renewable Share (%)',
        x=regional_stats.index,
        y=regional_stats['Renewable energy share in the total final energy consumption (%)'],
        marker_color='#2ca02c',
        hovertemplate='<b>%{x}</b><br>Renewable Share: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Regional SDG-7 Progress Comparison ({year}) - Select regions below to see country details",
        xaxis_title="Region",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400,
        xaxis={'tickangle': 45},
        clickmode='event+select'
    )
    
    return fig, year_data

def create_country_detail_chart(df_with_regions, selected_regions, year):
    """Create detailed country breakdown for selected regions"""
    if not selected_regions:
        # Show top 15 countries with lowest electricity access if no regions selected
        country_data = df_with_regions.dropna(subset=['Access to electricity (% of population)'])
        top_countries = country_data.nsmallest(15, 'Access to electricity (% of population)')
        title_suffix = "Countries with Lowest Electricity Access"
    else:
        # Filter to selected regions
        country_data = df_with_regions[df_with_regions['Region'].isin(selected_regions)]
        top_countries = country_data.dropna(subset=['Access to electricity (% of population)']).head(20)
        regions_text = ", ".join(selected_regions)
        title_suffix = f"Countries in {regions_text}"
    
    if len(top_countries) == 0:
        return None
    
    # Create scatter plot
    fig = go.Figure()
    
    # Color by region
    for region in top_countries['Region'].unique():
        region_data = top_countries[top_countries['Region'] == region]
        
        fig.add_trace(go.Scatter(
            x=region_data['Access to electricity (% of population)'],
            y=region_data['Access to clean fuels for cooking'],
            mode='markers+text',
            name=region,
            text=region_data['Entity'],
            textposition='top center',
            marker=dict(
                size=10,
                opacity=0.7
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Electricity Access: %{x:.1f}%<br>' +
                         'Clean Cooking Access: %{y:.1f}%<br>' +
                         'Region: ' + region + '<extra></extra>'
        ))
    
    # Add quadrant lines
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=75, y=75, text="High Access<br>Both Services", 
                      showarrow=False, font=dict(color="green", size=10), opacity=0.7)
    fig.add_annotation(x=25, y=25, text="Low Access<br>Both Services", 
                      showarrow=False, font=dict(color="red", size=10), opacity=0.7)
    
    fig.update_layout(
        title=f"Country-Level SDG-7 Access ({year}) - {title_suffix}",
        xaxis_title="Electricity Access (% of population)",
        yaxis_title="Clean Cooking Access (% of population)",
        height=400,
        xaxis=dict(range=[0, 105]),
        yaxis=dict(range=[0, 105]),
        showlegend=True
    )
    
    return fig

def main():
    # Load data
    df = load_and_clean_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Year selector
    years = sorted(df['Year'].unique())
    selected_year = st.sidebar.selectbox(
        "Select Year",
        years,
        index=len(years)-1  # Default to latest year
    )
    
    # Metric selector
    metric_options = {
        'total': 'Total Emissions',
        'per_gdp': 'Emissions per GDP',
        'per_capita': 'Emission Intensity (proxy)'
    }
    selected_metric = st.sidebar.selectbox(
        "Emission Metric",
        list(metric_options.keys()),
        format_func=lambda x: metric_options[x]
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Global map
        st.subheader("🗺️ Global Emission Intensity Map")
        map_fig = create_emissions_map(df, selected_year, selected_metric)
        st.plotly_chart(map_fig, use_container_width=True)
    
    with col2:
        # Top emitters chart
        st.subheader("📊 Top Emitters")
        top_emitters_fig = create_top_emitters_chart(df, selected_year)
        st.plotly_chart(top_emitters_fig, use_container_width=True)
    
    # Key insights section
    st.subheader("🔍 Key Insights")
    
    year_data = df[df['Year'] == selected_year]
    
    # Handle NaN values in calculations
    total_emissions = year_data['Value_co2_emissions_kt_by_country'].sum(skipna=True)
    avg_access = year_data['Access to electricity (% of population)'].mean(skipna=True)
    
    # Calculate high emitters only from non-NaN values
    co2_data = year_data.dropna(subset=['Value_co2_emissions_kt_by_country'])
    if len(co2_data) > 0:
        high_emitters = len(co2_data[co2_data['Value_co2_emissions_kt_by_country'] > co2_data['Value_co2_emissions_kt_by_country'].quantile(0.8)])
    else:
        high_emitters = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Global CO₂ Emissions",
            f"{total_emissions:,.0f} kt",
            delta=None
        )
    
    with col2:
        st.metric(
            "Avg. Electricity Access",
            f"{avg_access:.1f}%" if not pd.isna(avg_access) else "No data",
            delta=None
        )
    
    with col3:
        st.metric(
            "High Emitters",
            f"{high_emitters} countries",
            help="Countries in top 20% of emissions"
        )
    
    with col4:
        # Handle case where CO2 data might have NaN values
        co2_data = year_data.dropna(subset=['Value_co2_emissions_kt_by_country'])
        if len(co2_data) > 0:
            highest_emitter = co2_data.loc[co2_data['Value_co2_emissions_kt_by_country'].idxmax(), 'Entity']
        else:
            highest_emitter = "No data"
        
        st.metric(
            "Highest Emitter",
            highest_emitter,
            delta=None
        )
    
    # Data insights
    st.markdown("""
    ### 💡 Understanding the Data
    
    - **Dark red regions** show countries with the highest emission intensity
    - **Developed nations** typically show higher per-capita emissions despite smaller populations
    - **Developing countries** often have lower total emissions but may be rapidly increasing
    - **Energy access gaps** correlate strongly with development status and future emission potential
    
    This visualization reveals the fundamental inequality in climate responsibility and highlights 
    countries that may need targeted intervention strategies.
    """)

    st.title("🎯 SDG-7 Progress Tracker")
    st.markdown("""
    **Tracking progress toward universal energy access with interactive coordinated visualizations**
    
    Sustainable Development Goal 7 aims to ensure access to affordable, reliable, sustainable and modern energy for all.
    This dashboard reveals how far countries remain from achieving universal energy access, featuring coordinated 
    regional and country-level analysis.
    """)
    
    # Load data
    df = load_and_clean_data()
    
    # Sidebar controls
    st.sidebar.header("🎯 SDG-7 Analysis Controls")
    
    sdg_year = st.sidebar.selectbox(
        "Select Year for Analysis",
        sorted(df['Year'].unique()),
        index=len(sorted(df['Year'].unique()))-1
    )
    
    # Define SDG-7 indicators
    sdg_indicators = {
        'electricity_access': 'Access to electricity (% of population)',
        'clean_cooking': 'Access to clean fuels for cooking',
        'renewable_share': 'Renewable energy share in the total final energy consumption (%)'
    }
    
    # Get year data for analysis
    sdg_year_data = df[df['Year'] == sdg_year]
    
    # ===== SDG-7 SUMMARY METRICS =====
    st.subheader("📊 Global SDG-7 Progress Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Global electricity access
        global_elec_access = sdg_year_data['Access to electricity (% of population)'].mean(skipna=True)
        st.metric(
            "Global Avg Electricity Access",
            f"{global_elec_access:.1f}%" if pd.notna(global_elec_access) else "No data",
            delta=None
        )
    
    with col2:
        # Global clean cooking access
        global_cooking_access = sdg_year_data['Access to clean fuels for cooking'].mean(skipna=True)
        st.metric(
            "Global Avg Clean Cooking",
            f"{global_cooking_access:.1f}%" if pd.notna(global_cooking_access) else "No data",
            delta=None
        )
    
    with col3:
        # Countries with universal electricity access
        universal_elec = len(sdg_year_data[sdg_year_data['Access to electricity (% of population)'] >= 99])
        st.metric(
            "Universal Electricity Access",
            f"{universal_elec} countries",
            delta=None
        )
    
    with col4:
        # Countries needing urgent support
        urgent_support = len(sdg_year_data[
            (sdg_year_data['Access to electricity (% of population)'] < 50) |
            (sdg_year_data['Access to clean fuels for cooking'] < 50)
        ])
        st.metric(
            "Need Urgent Support",
            f"{urgent_support} countries",
            help="Countries with <50% access to electricity or clean cooking"
        )
    
    # ===== MAIN VISUALIZATIONS =====
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ SDG-7 Global Progress Map")
        
        # Indicator selector
        selected_indicator = st.selectbox(
            "Select SDG-7 Indicator to Display:",
            list(sdg_indicators.keys()),
            format_func=lambda x: sdg_indicators[x]
        )
        
        sdg_map_fig = create_sdg_map(df, sdg_year, sdg_indicators[selected_indicator])
        if sdg_map_fig:
            st.plotly_chart(sdg_map_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Largest Access Gaps")
        gap_analysis_fig = create_sdg_gap_analysis(df, sdg_year)
        if gap_analysis_fig:
            st.plotly_chart(gap_analysis_fig, use_container_width=True)
    
    # ===== REGIONAL COMPARISON WITH COORDINATED VISUALIZATIONS =====
    st.subheader("🌍 Regional SDG-7 Comparison & Country Details")
    
    # Create regional comparison chart and get data with regions
    regional_fig, df_with_regions = create_regional_comparison(df, sdg_year)
    
    if regional_fig and df_with_regions is not None:
        # Region selector for coordination
        available_regions = df_with_regions['Region'].unique()
        available_regions = [r for r in available_regions if r != 'Other']
        
        selected_regions_coord = st.multiselect(
            "Select regions to view country details (or leave empty to see countries with lowest access):",
            options=sorted(available_regions),
            default=[],
            key="coord_regions"
        )
        
        # Display coordinated visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(regional_fig, use_container_width=True)
            if not selected_regions_coord:
                st.info("💡 Select regions above to see detailed country breakdowns, or view countries with lowest access by default")
        
        with col2:
            country_detail_fig = create_country_detail_chart(df_with_regions, selected_regions_coord, sdg_year)
            if country_detail_fig:
                st.plotly_chart(country_detail_fig, use_container_width=True)
            else:
                st.info("No country data available for selected regions")
        
        # Explanation of coordination
        with st.expander("📊 How to Use These Coordinated Charts"):
            st.markdown("""
            **Regional Overview (Left)**: Shows average SDG-7 indicators by region
            **Country Details (Right)**: Shows individual countries within selected regions
            
            **How to Interact:**
            1. **Select regions** using the dropdown above to see specific countries in those regions
            2. **Leave empty** to see the 15 countries with lowest electricity access globally
            3. **Compare regions** in the left chart to identify best/worst performing areas
            4. **Examine countries** in the right chart to see which specific nations need support
            
            **Reading the Country Detail Chart:**
            - **X-axis**: Electricity access percentage
            - **Y-axis**: Clean cooking access percentage  
            - **Top-right quadrant**: Countries with high access to both
            - **Bottom-left quadrant**: Countries needing urgent support for both
            - **Country labels**: Hover or look at text labels for country names
            """)
    else:
        st.info("Regional comparison data not available for selected year")
    
    # ===== PROGRESS TRACKING =====
    st.subheader("📈 SDG-7 Progress Tracking")
    
    # Country selector for progress tracking
    countries_with_data = df.dropna(subset=['Access to electricity (% of population)'])['Entity'].unique()
    selected_countries_sdg = st.multiselect(
        "Select countries to track SDG-7 progress:",
        options=sorted(countries_with_data),
        default=['Nigeria', 'India', 'Bangladesh', 'Ethiopia', 'Kenya'] if all(c in countries_with_data for c in ['Nigeria', 'India', 'Bangladesh', 'Ethiopia', 'Kenya']) else list(countries_with_data)[:5]
    )
    
    if selected_countries_sdg:
        # Create progress charts for electricity access
        progress_fig = create_sdg_progress_chart(df, selected_countries_sdg, 'Access to electricity (% of population)')
        if progress_fig:
            st.plotly_chart(progress_fig, use_container_width=True)
    
    # ===== KEY INSIGHTS =====
    st.subheader("💡 SDG-7 Key Insights")
    
    # Countries needing urgent support
    urgent_countries = sdg_year_data[
        (sdg_year_data['Access to electricity (% of population)'] < 50) |
        (sdg_year_data['Access to clean fuels for cooking'] < 50)
    ]
    
    if len(urgent_countries) > 0:
        st.error(f"**{len(urgent_countries)} countries** need urgent energy access support, top countries are:")
        for _, country in urgent_countries.head(8).iterrows():
            elec_access = country['Access to electricity (% of population)']
            cooking_access = country['Access to clean fuels for cooking']
            gaps = []
            if pd.notna(elec_access) and elec_access < 50:
                gaps.append(f"Electricity: {elec_access:.1f}%")
            if pd.notna(cooking_access) and cooking_access < 50:
                gaps.append(f"Clean cooking: {cooking_access:.1f}%")
            st.write(f"• **{country['Entity']}**: {', '.join(gaps)}")
    
    # Progress champions
    champions = sdg_year_data[
        (sdg_year_data['Access to electricity (% of population)'] >= 99) &
        (sdg_year_data['Access to clean fuels for cooking'] >= 80)
    ]
    
    if len(champions) > 0:
        st.success(f"**{len(champions)} countries** are SDG-7 champions with near-universal access, top countries are:")
        for _, country in champions.head(8).iterrows():
            elec_access = country['Access to electricity (% of population)']
            cooking_access = country['Access to clean fuels for cooking']
            st.write(f"• **{country['Entity']}**: Electricity {elec_access:.1f}%, Clean cooking {cooking_access:.1f}%")
    
    # ===== EDUCATIONAL CONTENT =====
    with st.expander("🎯 Understanding SDG-7"):
        st.markdown("""
        **Sustainable Development Goal 7: Affordable and Clean Energy**
        
        **Key Targets:**
        - **7.1**: Universal access to affordable, reliable, and modern energy services
        - **7.2**: Increase substantially the share of renewable energy in the global energy mix
        - **7.3**: Double the global rate of improvement in energy efficiency
        
        **Global Context:**
        - **2.8 billion people** still lack access to clean cooking solutions
        - **759 million people** still lack access to electricity 
        - Energy poverty disproportionately affects rural areas and developing countries
        - Clean energy access is fundamental to health, education, and economic development
        
        **Dashboard Insights:**
        - **Red regions** show countries furthest from SDG-7 targets
        - **Green regions** have achieved or are close to universal access
        - **Gap analysis** identifies priority countries for targeted support
        - **Progress tracking** shows improvement trends over time
        
        **Why SDG-7 Matters:**
        - Access to electricity enables education, healthcare, and economic opportunities
        - Clean cooking prevents indoor air pollution that kills 3.8 million annually
        - Renewable energy supports climate goals while expanding access
        - Energy infrastructure is foundational for achieving other SDGs
        """)
    
    
    # Footer
    st.markdown("---")
    st.markdown("*Data source: Global Data on Sustainable Energy | Dashboard created for climate policy analysis*")

if __name__ == "__main__":
    main()
