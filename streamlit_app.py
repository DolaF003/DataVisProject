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
st.title("🌍 Two Worlds, One Planet: A Global Energy & Progress Dashboard")
st.markdown("""
**Exploring the uneven responsibility for climate change across nations**

This dashboard reveals the stark disparities in carbon emissions between developed and developing countries,
highlighting how some nations contribute disproportionately to the climate crisis while others struggle
with basic energy access.
""")

def add_regional_mapping(df):
    """Add regional mapping to the dataframe"""
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
        'South Africa': 'Sub-Saharan Africa',
        'Zimbabwe': 'Sub-Saharan Africa',
        'Zambia': 'Sub-Saharan Africa',
        'Botswana': 'Sub-Saharan Africa',
        'Namibia': 'Sub-Saharan Africa',
        'Angola': 'Sub-Saharan Africa',
        'Cameroon': 'Sub-Saharan Africa',
        'Ivory Coast': 'Sub-Saharan Africa',
        'Democratic Republic of Congo': 'Sub-Saharan Africa',
        'Congo': 'Sub-Saharan Africa',
        'Gabon': 'Sub-Saharan Africa',
        'Central African Republic': 'Sub-Saharan Africa',
        'Sudan': 'Sub-Saharan Africa',
        'South Sudan': 'Sub-Saharan Africa',
        'Somalia': 'Sub-Saharan Africa',
        'Eritrea': 'Sub-Saharan Africa',
        'Djibouti': 'Sub-Saharan Africa',
        'Malawi': 'Sub-Saharan Africa',
        'Lesotho': 'Sub-Saharan Africa',
        'Swaziland': 'Sub-Saharan Africa',
        'Mauritius': 'Sub-Saharan Africa',
        'Seychelles': 'Sub-Saharan Africa',
        'India': 'South Asia',
        'Bangladesh': 'South Asia',
        'Pakistan': 'South Asia',
        'Afghanistan': 'South Asia',
        'Nepal': 'South Asia',
        'Sri Lanka': 'South Asia',
        'Myanmar': 'South Asia',
        'Bhutan': 'South Asia',
        'Maldives': 'South Asia',
        'China': 'East Asia',
        'Mongolia': 'East Asia',
        'North Korea': 'East Asia',
        'South Korea': 'East Asia',
        'Japan': 'East Asia',
        'Taiwan': 'East Asia',
        'Hong Kong': 'East Asia',
        'Macau': 'East Asia',
        'Indonesia': 'Southeast Asia',
        'Philippines': 'Southeast Asia',
        'Vietnam': 'Southeast Asia',
        'Cambodia': 'Southeast Asia',
        'Laos': 'Southeast Asia',
        'Thailand': 'Southeast Asia',
        'Malaysia': 'Southeast Asia',
        'Singapore': 'Southeast Asia',
        'Brunei': 'Southeast Asia',
        'Timor-Leste': 'Southeast Asia',
        'Brazil': 'Latin America',
        'Peru': 'Latin America',
        'Bolivia': 'Latin America',
        'Guatemala': 'Latin America',
        'Haiti': 'Latin America',
        'Honduras': 'Latin America',
        'Nicaragua': 'Latin America',
        'Paraguay': 'Latin America',
        'Ecuador': 'Latin America',
        'Colombia': 'Latin America',
        'Venezuela': 'Latin America',
        'Argentina': 'Latin America',
        'Chile': 'Latin America',
        'Uruguay': 'Latin America',
        'Guyana': 'Latin America',
        'Suriname': 'Latin America',
        'French Guiana': 'Latin America',
        'Costa Rica': 'Latin America',
        'Panama': 'Latin America',
        'El Salvador': 'Latin America',
        'Belize': 'Latin America',
        'Dominican Republic': 'Latin America',
        'Jamaica': 'Latin America',
        'Cuba': 'Latin America',
        'Trinidad and Tobago': 'Latin America',
        'Barbados': 'Latin America',
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
        'Greece': 'Europe',
        'Cyprus': 'Europe',
        'Malta': 'Europe',
        'Luxembourg': 'Europe',
        'Iceland': 'Europe',
        'Montenegro': 'Europe',
        'Kosovo': 'Europe',
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
        'Israel': 'Middle East & North Africa',
        'Palestine': 'Middle East & North Africa',
        'Kuwait': 'Middle East & North Africa',
        'Qatar': 'Middle East & North Africa',
        'United Arab Emirates': 'Middle East & North Africa',
        'Bahrain': 'Middle East & North Africa',
        'Oman': 'Middle East & North Africa',
        'Australia': 'Oceania',
        'New Zealand': 'Oceania',
        'Papua New Guinea': 'Oceania',
        'Fiji': 'Oceania',
        'Solomon Islands': 'Oceania',
        'Vanuatu': 'Oceania',
        'Samoa': 'Oceania',
        'Tonga': 'Oceania',
        'Kiribati': 'Oceania',
        'Palau': 'Oceania',
        'Marshall Islands': 'Oceania',
        'Micronesia': 'Oceania',
        'Nauru': 'Oceania',
        'Tuvalu': 'Oceania'
    }
    
    # Add region column
    df['Region'] = df['Entity'].map(region_mapping).fillna('Other')
    return df

def create_regional_filter_controls():
    """Create regional filter controls for the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("🌍 Regional Filters")
    
    # Define all available regions
    all_regions = [
        'Sub-Saharan Africa',
        'South Asia', 
        'East Asia',
        'Southeast Asia',
        'Latin America',
        'North America',
        'Europe',
        'Middle East & North Africa',
        'Oceania'
    ]
    
    # Region selection
    filter_regions = st.sidebar.multiselect(
        "Select Regions to Focus On:",
        options=['All Regions'] + all_regions,
        default=['All Regions'],
        help="Select specific regions to filter the analysis. 'All Regions' shows global data.",
        key="regional_filter_multiselect"
    )
    
    # Convert 'All Regions' selection to actual region list
    if 'All Regions' in filter_regions or len(filter_regions) == 0:
        selected_regions = all_regions
        show_all = True
    else:
        selected_regions = filter_regions
        show_all = False
    
    # Show region summary
    if not show_all:
        st.sidebar.info(f"Filtering data for: {', '.join(selected_regions)}")
    
    return selected_regions, show_all

def filter_data_by_regions(df, selected_regions, show_all=False):
    """Filter dataframe by selected regions"""
    if show_all:
        return df
    else:
        # Filter out 'Other' region and apply regional filter
        filtered_df = df[df['Region'].isin(selected_regions)]
        return filtered_df

def create_regional_summary_metrics(df, selected_regions, year, show_all):
    """Create summary metrics for selected regions"""
    # Filter data for the year and regions
    year_data = df[df['Year'] == year]
    if not show_all:
        year_data = year_data[year_data['Region'].isin(selected_regions)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Number of countries in selection
        num_countries = len(year_data['Entity'].unique())
        st.metric(
            "Countries in Selection",
            f"{num_countries}",
            delta=None
        )
    
    with col2:
        # Average emissions for selected regions
        avg_emissions = year_data['Value_co2_emissions_kt_by_country'].mean(skipna=True)
        st.metric(
            "Avg CO₂ Emissions",
            f"{avg_emissions:,.0f} kt" if pd.notna(avg_emissions) else "No data",
            delta=None
        )
    
    with col3:
        # Regional electricity access average
        avg_elec_access = year_data['Access to electricity (% of population)'].mean(skipna=True)
        st.metric(
            "Avg Electricity Access", 
            f"{avg_elec_access:.1f}%" if pd.notna(avg_elec_access) else "No data",
            delta=None
        )
    
    with col4:
        # Number of regions selected
        regions_count = len(selected_regions) if not show_all else 9
        st.metric(
            "Regions in Analysis",
            f"{regions_count}",
            delta=None
        )

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
    
    df = df[df['Year'] != 2020]

    # Add regional mapping
    df = add_regional_mapping(df)
    
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
        title=f"Regional SDG-7 Progress Comparison ({year})",
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
    
    # Regional filtering controls
    selected_regions, show_all = create_regional_filter_controls()
    
    # Apply regional filtering to the dataset
    df_filtered = filter_data_by_regions(df, selected_regions, show_all)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Year selector
    years = sorted(df_filtered['Year'].unique())

    # Default to 2019 for carbon emissions analysis, or latest available if 2019 not available
    if 2019 in years:
        default_index = years.index(2019)
    else:
        default_index = len(years)-1 if len(years) > 0 else 0

    selected_year = st.sidebar.selectbox(
        "Select Year",
        years,
        index=default_index,
        key="main_year_selector"
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
        format_func=lambda x: metric_options[x],
        key="emission_metric_selector"
    )
    
    # Add regional summary metrics section
    if not show_all:
        with st.expander("📊 Regional Summary", expanded=True):
            create_regional_summary_metrics(df_filtered, selected_regions, selected_year, show_all)
    
    # Create tabs for main content organization
    tab1, tab2 = st.tabs(["🏭 Carbon Emissions Analysis", "🎯 SDG-7 Progress Analysis"])
    
    # ===== TAB 1: CARBON EMISSIONS ANALYSIS =====
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Global Emission Intensity Map")
            
            # Instructions for Global Emissions Map
            with st.expander("📋 How to Use This Chart"):
                st.markdown("""
                **Step-by-Step Instructions:**
                1. **Select Year**: Use the sidebar "Select Year" dropdown to choose which year to analyze
                2. **Choose Metric**: Use the sidebar "Emission Metric" dropdown to switch between:
                   - *Total Emissions*: See absolute CO₂ emissions by country
                   - *Emissions per GDP*: See emissions relative to economic output
                   - *Emission Intensity*: See a simplified intensity measure
                3. **Regional Focus**: Use the sidebar "Regional Filters" to focus on specific world regions
                4. **Explore the Map**: 
                   - Hover over countries to see detailed emission data
                   - Darker red = higher emissions/intensity
                   - Grey areas = no data available
                5. **Analyze Patterns**: Look for regional clusters and compare developed vs. developing nations
                """)
            
            map_fig = create_emissions_map(df_filtered, selected_year, selected_metric)
            st.plotly_chart(map_fig, use_container_width=True)
        
        with col2:
            st.subheader("🏭 Top Emitters")
            
            # Instructions for Top Emitters Chart
            with st.expander("📋 How to Use This Chart"):
                st.markdown("""
                **Step-by-Step Instructions:**
                1. **View Rankings**: This chart automatically shows the top 10 highest emitters for your selected year
                2. **Change Year**: Use the sidebar "Select Year" to see rankings for different years
                3. **Regional Filter**: Use sidebar "Regional Filters" to see top emitters within specific regions only
                4. **Read the Data**: 
                   - Horizontal bars show CO₂ emissions in kilotons (kt)
                   - Countries are ranked from highest (top) to lowest (bottom)
                   - Color intensity corresponds to emission levels
                5. **Compare**: Notice how rankings change across years and regions
                """)
            
            top_emitters_fig = create_top_emitters_chart(df_filtered, selected_year)
            st.plotly_chart(top_emitters_fig, use_container_width=True)
        
        # Key insights section for carbon emissions
        with st.expander("📈 Carbon Emissions Key Insights"):
            year_data = df_filtered[df_filtered['Year'] == selected_year]
            
            # Handle NaN values in calculations
            total_emissions = year_data['Value_co2_emissions_kt_by_country'].sum(skipna=True)
            avg_access = year_data['Access to electricity (% of population)'].mean(skipna=True)
            
            # Calculate high emitters only from non-NaN values
            co2_data = year_data.dropna(subset=['Value_co2_emissions_kt_by_country'])
            if len(co2_data) > 0:
                high_emitters = len(co2_data[co2_data['Value_co2_emissions_kt_by_country'] > co2_data['Value_co2_emissions_kt_by_country'].quantile(0.8)])
                highest_emitter = co2_data.loc[co2_data['Value_co2_emissions_kt_by_country'].idxmax(), 'Entity']
            else:
                high_emitters = 0
                highest_emitter = "No data"
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total CO₂ Emissions",
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
                st.metric(
                    "Highest Emitter",
                    highest_emitter,
                    delta=None
                )
            
            st.markdown("""
            **Key Patterns:**
            - **Dark red regions** show countries with the highest emission intensity
            - **Developed nations** typically show higher per-capita emissions despite smaller populations
            - **Developing countries** often have lower total emissions but may be rapidly increasing
            - **Energy access gaps** correlate strongly with development status and future emission potential
            """)
    
    # ===== TAB 2: SDG-7 PROGRESS ANALYSIS =====
    with tab2:
        # SDG Year selector in tab
        sdg_year = st.selectbox(
            "Select Year for SDG-7 Analysis:",
            sorted(df_filtered['Year'].unique()),
            index=len(sorted(df_filtered['Year'].unique()))-1,
            key="sdg_year_selector"
        )
        
        # Define SDG-7 indicators
        sdg_indicators = {
            'electricity_access': 'Access to electricity (% of population)',
            'clean_cooking': 'Access to clean fuels for cooking',
            'renewable_share': 'Renewable energy share in the total final energy consumption (%)'
        }
        
        # Get year data for analysis
        sdg_year_data = df_filtered[df_filtered['Year'] == sdg_year]
        
        # SDG-7 Summary Metrics
        with st.expander("📊 SDG-7 Global Progress Summary", expanded=True):
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
        
        # Main SDG-7 Visualizations
        st.subheader("🗺️ SDG-7 Global Progress Map")
        
        # Instructions for SDG-7 Map
        with st.expander("📋 How to Use This Chart"):
            st.markdown("""
            **Step-by-Step Instructions:**
            1. **Select Year**: Use the dropdown above to choose which year to examine
            2. **Choose Indicator**: Use the dropdown below to select which SDG-7 indicator to display:
               - *Electricity Access*: % of population with electricity access
               - *Clean Cooking*: % of population with clean cooking access  
               - *Renewable Share*: % of renewable energy in total consumption
            3. **Regional Focus**: Use sidebar "Regional Filters" to focus on specific world regions
            4. **Interpret Colors**:
               - Green = High progress toward SDG-7 targets
               - Red = Low progress, needs urgent attention
               - Grey = No data available
            5. **Explore**: Hover over countries to see exact percentages and identify priority areas
            """)
        
        # Indicator selector
        selected_indicator = st.selectbox(
            "Select SDG-7 Indicator to Display:",
            list(sdg_indicators.keys()),
            format_func=lambda x: sdg_indicators[x],
            key="sdg_indicator_selector"
        )
        
        sdg_map_fig = create_sdg_map(df_filtered, sdg_year, sdg_indicators[selected_indicator])
        if sdg_map_fig:
            st.plotly_chart(sdg_map_fig, use_container_width=True)
        
        # Largest Access Gaps chart below the map
        st.subheader("📊 Largest Access Gaps")
        
        # Instructions for Access Gaps Chart
        with st.expander("📋 How to Use This Chart"):
            st.markdown("""
            **Step-by-Step Instructions:**
            1. **Automatic Display**: This chart automatically shows the 15 countries with the largest gaps to universal access
            2. **Change Year**: Use the dropdown above to see gaps for different years
            3. **Regional Filter**: Use sidebar "Regional Filters" to focus on gaps within specific regions
            4. **Read the Stacked Bars**:
               - Red bars = Gap in electricity access (% away from 100%)
               - Teal bars = Gap in clean cooking access (% away from 100%)
               - Taller bars = Countries needing more urgent support
            5. **Identify Priorities**: Countries at the top need the most comprehensive energy access interventions
            """)
        
        gap_analysis_fig = create_sdg_gap_analysis(df_filtered, sdg_year)
        if gap_analysis_fig:
            st.plotly_chart(gap_analysis_fig, use_container_width=True)
        
        # Regional Comparison and Country Details
        st.subheader("🌍 Regional SDG-7 Comparison & Country Details")
        
        # Instructions for Regional Comparison
        with st.expander("📋 How to Use These Coordinated Charts"):
            st.markdown("""
            **Step-by-Step Instructions:**
            1. **Understand the Layout**:
               - Left chart = Regional averages for SDG-7 indicators
               - Right chart = Individual country details within selected regions
            2. **Select Regions**: Use the multiselect dropdown below to choose which regions to examine
               - Leave empty to see the 15 countries with lowest electricity access globally
               - Select one or more regions to see countries within those regions
            3. **Read the Regional Chart (Left)**:
               - Blue bars = Average electricity access by region
               - Orange bars = Average clean cooking access by region  
               - Green bars = Average renewable energy share by region
            4. **Interpret Country Details (Right)**:
               - X-axis = Electricity access %
               - Y-axis = Clean cooking access %
               - Top-right = Countries with high access to both services
               - Bottom-left = Countries needing urgent support for both
            5. **Interactive Analysis**: Compare regions first, then drill down to see which specific countries need attention
            """)
        
        # Create regional comparison chart and get data with regions
        regional_fig, df_with_regions = create_regional_comparison(df_filtered, sdg_year)
        
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
            
            with col2:
                country_detail_fig = create_country_detail_chart(df_with_regions, selected_regions_coord, sdg_year)
                if country_detail_fig:
                    st.plotly_chart(country_detail_fig, use_container_width=True)
                else:
                    st.info("No country data available for selected regions")
        else:
            st.info("Regional comparison data not available for selected year")
        
        # Progress Tracking
        st.subheader("📈 SDG-7 Progress Tracking")
        
        # Instructions for Progress Tracking
        with st.expander("📋 How to Use This Chart"):
            st.markdown("""
            **Step-by-Step Instructions:**
            1. **Select Countries**: Use the multiselect dropdown below to choose which countries to track over time
               - Default selection includes key developing countries
               - Add or remove countries to customize your analysis
            2. **Regional Filter**: Use sidebar "Regional Filters" to limit country options to specific regions
            3. **Read the Line Chart**:
               - X-axis = Years (time progression)
               - Y-axis = Electricity access percentage
               - Each colored line = One country's progress over time
               - Green dashed line = SDG target (100% access)
            4. **Analyze Trends**:
               - Steep upward lines = Rapid progress
               - Flat lines = Stagnant progress
               - Lines approaching 100% = Near universal access
            5. **Compare Performance**: See which countries are making fastest progress toward SDG-7 targets
            """)
        
        # Country selector for progress tracking
        countries_with_data = df_filtered.dropna(subset=['Access to electricity (% of population)'])['Entity'].unique()
        selected_countries_sdg = st.multiselect(
            "Select countries to track SDG-7 progress:",
            options=sorted(countries_with_data),
            default=['Nigeria', 'India', 'Bangladesh', 'Ethiopia', 'Kenya'] if all(c in countries_with_data for c in ['Nigeria', 'India', 'Bangladesh', 'Ethiopia', 'Kenya']) else list(countries_with_data)[:5],
            key="sdg_countries_progress_selector"
        )
        
        if selected_countries_sdg:
            # Create progress charts for electricity access
            progress_fig = create_sdg_progress_chart(df_filtered, selected_countries_sdg, 'Access to electricity (% of population)')
            if progress_fig:
                st.plotly_chart(progress_fig, use_container_width=True)
        
        # Key Insights for SDG-7
        with st.expander("🔍 SDG-7 Key Insights"):
            # Countries needing urgent support
            urgent_countries = sdg_year_data[
                (sdg_year_data['Access to electricity (% of population)'] < 50) |
                (sdg_year_data['Access to clean fuels for cooking'] < 50)
            ]
            
            if len(urgent_countries) > 0:
                st.error(f"**{len(urgent_countries)} countries** need urgent energy access support:")
                for _, country in urgent_countries.head(5).iterrows():
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
                st.success(f"**{len(champions)} countries** are SDG-7 champions with near-universal access:")
                for _, country in champions.head(5).iterrows():
                    elec_access = country['Access to electricity (% of population)']
                    cooking_access = country['Access to clean fuels for cooking']
                    st.write(f"• **{country['Entity']}**: Electricity {elec_access:.1f}%, Clean cooking {cooking_access:.1f}%")
    
    
        with st.expander("📚 Understanding the Data & Methodology"):
            st.markdown("""
            ### 🎯 **SDG-7: Affordable and Clean Energy**
        
            **Key Targets:**
            - **Target 7.1**: Universal access to affordable, reliable, and modern energy services
            - **Target 7.2**: Increase substantially the share of renewable energy in the global energy mix
            - **Target 7.3**: Double the global rate of improvement in energy efficiency
        
            **Global Context:**
            - **2.8 billion people** still lack access to clean cooking solutions
            - **759 million people** still lack access to electricity
            - Energy poverty disproportionately affects rural areas and developing countries
            - Clean energy access is fundamental to health, education, and economic development
        
            **Why SDG-7 Matters:**
            - Access to electricity enables education, healthcare, and economic opportunities
            - Clean cooking prevents indoor air pollution that kills 3.8 million annually
            - Renewable energy supports climate goals while expanding access
            - Energy infrastructure is foundational for achieving other SDGs
        
            **Data Source:** Global Data on Sustainable Energy | Dashboard created for climate policy analysis
            """)

if __name__ == "__main__":
    main()
