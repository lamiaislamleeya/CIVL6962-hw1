import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Transportation Dashboard", layout="wide")
st.title("Transportation Data Dashboard")
st.write("CIVL 6962 · HW1")

# CACHE: Load data once, reuse every time
@st.cache_data
def load_data():
    return pd.read_parquet("pems.parquet")

df = load_data()

# ===== SIDEBAR: NAVIGATION MENU ONLY =====
with st.sidebar:
    st.header("📑 Menu")
    page = st.radio("Select a page", [
        "📊 Charts",
        "📍 Data Provenance",
        "⚠️ Blind Spots"
    ])

# ===== CONTROLS =====
if page == "📊 Charts":
    st.subheader("📊 Controls")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Control 1: Select a sensor
        sensor = st.selectbox("Sensor", sorted(df['sensor'].unique()))

    with col2:
        # Control 2: Select a metric
        metric = st.radio("Metric", ["flow", "occupancy", "speed"])

    with col3:
        # Control 3: Days of history to show
        days = st.slider("Days of history", 1, 60, 7)

    st.markdown("---")
else:
    sensor = df['sensor'].unique()[0]
    metric = "flow"
    days = 7

filtered_df = df[df['sensor'] == sensor].copy()
max_date = filtered_df['time'].max()
min_date = max_date - pd.Timedelta(days=days)
filtered_df = filtered_df[filtered_df['time'] >= min_date]

# Set units for display
units = {"flow": "(vehicles/5min)", "occupancy": "(%)", "speed": "(mph)"}
unit_label = units[metric]

# ===== MAIN AREA =====

# PAGE 1: CHARTS
if page == "📊 Charts":
    st.subheader(f"📈 Key Metrics - {sensor} ({metric.capitalize()}) {unit_label}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{filtered_df[metric].mean():.1f}")
    col2.metric("Max", f"{filtered_df[metric].max():.1f}")
    col3.metric("Min", f"{filtered_df[metric].min():.1f}")
    col4.metric("Std Dev", f"{filtered_df[metric].std():.1f}")
    
    st.markdown("---")
    
    st.subheader(f"Charts - {sensor} ({metric.capitalize()}) {unit_label}")
    
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Time Series", "Hourly Average", "Heatmap"])
    
    # Chart 1: Line chart
    with chart_tab1:
        chart_data = filtered_df.set_index('time')[[metric]].rename(columns={metric: f"{metric} {unit_label}"})
        st.line_chart(chart_data)
        st.caption(f"Showing {len(filtered_df):,} 5-minute measurements")
    
    # Chart 2: Bar chart
    with chart_tab2:
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['hour'] = filtered_df_copy['time'].dt.hour
        hourly = filtered_df_copy.groupby('hour')[metric].mean().rename(f"{metric} {unit_label}")
        st.bar_chart(hourly)
        st.caption("Average by hour of day")
    
    # Chart 3: Heatmap
    with chart_tab3:
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['day_of_week'] = filtered_df_copy['time'].dt.day_name()
        filtered_df_copy['hour'] = filtered_df_copy['time'].dt.hour
        
        heatmap_data = filtered_df_copy.pivot_table(
            values=metric, 
            index='day_of_week', 
            columns='hour', 
            aggfunc='mean'
        )
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
        
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(heatmap_data, cmap='viridis', ax=ax, cbar_kws={'label': unit_label})
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        st.pyplot(fig, use_container_width=True)

# PAGE 2: DATA PROVENANCE
elif page == "📍 Data Provenance":
    st.subheader("Data Provenance")
    st.markdown("""
    **Who collected it**  
    California Department of Transportation (Caltrans) - PeMS Program
    
    **Where**  
    District 4 (San Francisco Bay Area freeway mainline)
    
    **When**  
    January 1 - February 28, 2018
    
    **With what instrument**  
    Inductive loop detectors embedded in freeway pavement
    
    **Time Resolution**  
    5-minute intervals (aggregated measurements)
    
    **Sensors included**  
    12 locations across District 4
    """)

# PAGE 3: BLIND SPOTS
elif page == "⚠️ Blind Spots":
    st.subheader("What This Dashboard CANNOT Tell About")
    
    st.markdown("""
    **1. Coverage Bias**  
    This dashboard shows only 12 mainline freeway sensors. A viewer might conclude that "Bay Area traffic follows these patterns". But arterial roads and local streets are completely invisible. These patterns do not represent overall regional traffic behavior.
    
    **2. Incident Invisibility**  
    Data is aggregated to 5-minute intervals. A viewer seeing smooth trend lines might believe "traffic is predictable and stable," but accidents or incidents lasting 1-3 minutes are averaged away. A major incident that causes 30 seconds of gridlock would be invisible on these charts.
    
    **3. Seasonal Bias**  
    Data is from January-February 2018. A viewer might assume these traffic patterns are normal throughout the year. But this dashboard cannot tell about traffic during typical work weeks in summer, spring, or fall.
    """)

# RAW DATA AT BOTTOM (always visible)
st.markdown("---")
st.subheader("📋 Raw Data")
st.write(f"Showing {len(filtered_df):,} records")
st.dataframe(filtered_df, use_container_width=True)