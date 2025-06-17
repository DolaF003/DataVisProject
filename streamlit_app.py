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
with basic energy access.
""")

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('global-data-on-sustainable-energy (1).csv')
    
    # Apply the cleaning steps we did earlier
    # Remove rows with missing Entity/Year
    df = df.dropna(subset=['Entity', 'Year'])
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('\n', ' ')
    
    # Remove rows with excessive missing values (>50%)
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)

    return df

def create_emissions_map(df, year, metric='total'):
    """Create choropleth map of emissions by country"""
    
    # Filter data for selected year
    year_data = df[df['Year'] == year].copy()
    
    if metric == 'total':
        color_col = 'Value_co2_emissions_kt_by_country'
        title = f"Total CO₂ Emissions by Country ({year})"
        color_label = "CO₂ Emissions (kt)"
    elif metric == 'per_gdp':
        # Calculate emissions per GDP
        year_data['emissions_per_gdp'] = year_data['Value_co2_emissions_kt_by_country'] / year_data['gdp_per_capita']
        color_col = 'emissions_per_gdp'
        title = f"CO₂ Emissions per GDP by Country ({year})"
        color_label = "CO₂ per GDP (kt per capita)"
    else:  # per_capita (simplified - would need population data)
        # For demo, use a proxy calculation
        year_data['emissions_proxy'] = year_data['Value_co2_emissions_kt_by_country'] / 1000  # Simplified
        color_col = 'emissions_proxy'
        title = f"CO₂ Emissions Intensity by Country ({year})"
        color_label = "Emission Intensity (proxy)"
    
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
    top_emitters = year_data.nlargest(top_n, 'Value_co2_emissions_kt_by_country')
    
    fig = px.bar(
        top_emitters,
        x='Value_co2_emissions_kt_by_country',
        y='Entity',
        orientation='h',
        title=f"Top {top_n} CO₂ Emitters ({year})",
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
    total_emissions = year_data['Value_co2_emissions_kt_by_country'].sum()
    avg_access = year_data['Access to electricity (% of population)'].mean()
    high_emitters = len(year_data[year_data['Value_co2_emissions_kt_by_country'] > year_data['Value_co2_emissions_kt_by_country'].quantile(0.8)])
    
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
            f"{avg_access:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "High Emitters",
            f"{high_emitters} countries",
            help="Countries in top 20% of emissions"
        )
    
    with col4:
        highest_emitter = year_data.loc[year_data['Value_co2_emissions_kt_by_country'].idxmax(), 'Entity']
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
    
    # Footer
    st.markdown("---")
    st.markdown("*Data source: Global Data on Sustainable Energy | Dashboard created for climate policy analysis*")

if __name__ == "__main__":
    main()


