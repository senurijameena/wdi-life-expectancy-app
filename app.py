# app.py — WDI Life Expectancy Dashboard


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


st.set_page_config(
    page_title = "🌍 WDI Life Expectancy Dashboard",
    page_icon  = "🌍",
    layout     = "wide"
)


@st.cache_data
def load_data():
    try:
        df_cluster    = pd.read_csv('wdi_clustering_ready.csv')
        df_regression = pd.read_csv('wdi_regression_ready.csv')
        return df_cluster, df_regression
    except FileNotFoundError:
        st.error("⚠️ Data files not found! Please run the main notebook first.")
        return None, None


@st.cache_resource
def load_models():
    try:
        model         = pickle.load(open('model.pkl', 'rb'))
        scaler        = pickle.load(open('scaler.pkl', 'rb'))
        features      = pickle.load(open('features.pkl', 'rb'))
        kmeans        = pickle.load(open('kmeans.pkl', 'rb'))
        scaler_clust  = pickle.load(open('scaler_cluster.pkl', 'rb'))
        clust_feats   = pickle.load(open('cluster_features.pkl', 'rb'))
        clust_names   = pickle.load(open('cluster_names.pkl', 'rb'))
        return model, scaler, features, kmeans, scaler_clust, clust_feats, clust_names
    except FileNotFoundError:
        return None, None, None, None, None, None, None

df_cluster, df_regression = load_data()
model, scaler, features, kmeans, scaler_clust, clust_feats, clust_names = load_models()

if df_cluster is None:
    st.stop()


st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/87/Color_icon_blue.png", width=30)
st.sidebar.title("🌍 WDI Dashboard")
st.sidebar.markdown("World Bank Development Indicators 2000–2025")
st.sidebar.divider()


page = st.sidebar.radio(
    "📌 Navigate",
    ["🏠 Overview", "📊 Explore Indicators", "🤖 Predict Life Expectancy", "🌍 Country Comparison"]
)

st.sidebar.divider()


regions = ["All"] + sorted(df_cluster['Region'].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("🗺️ Filter by Region", regions)

min_year = int(df_cluster['Year'].min()) if 'Year' in df_cluster.columns else 2022
max_year = int(df_cluster['Year'].max()) if 'Year' in df_cluster.columns else 2022

if min_year < max_year:
    selected_year = st.sidebar.slider("📅 Select Year", min_year, max_year, max_year)
else:
    selected_year = min_year
    st.sidebar.write(f"📅 Year: {min_year}")


df_year = df_cluster[df_cluster['Year'] == selected_year] if 'Year' in df_cluster.columns else df_cluster
if selected_region != "All":
    df_year = df_year[df_year['Region'] == selected_region]

if df_year.empty:
    st.warning(f"No data for {selected_region} in {selected_year}")
    st.stop()


# PAGE 1 — OVERVIEW

if page == "🏠 Overview":

    st.title("🌍 World Development Indicators Dashboard")
    st.caption(f"Exploring global development patterns | Year: {selected_year} | Region: {selected_region}")
    st.divider()

  
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌍 Countries",
            len(df_year)
        )
    with col2:
        avg_life = df_year['LifeExpectancy'].mean() if 'LifeExpectancy' in df_year.columns else 0
        st.metric(
            "❤️ Avg Life Expectancy",
            f"{avg_life:.1f} years"
        )
    with col3:
        avg_gdp = df_year['GDP_per_capita'].mean() if 'GDP_per_capita' in df_year.columns else 0
        st.metric(
            "💰 Avg GDP per Capita",
            f"${avg_gdp:,.0f}"
        )
    with col4:
        avg_infant = df_year['InfantMortality'].mean() if 'InfantMortality' in df_year.columns else 0
        st.metric(
            "👶 Avg Infant Mortality",
            f"{avg_infant:.1f} per 1000"
        )

    st.divider()

  
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"📊 Life Expectancy Distribution ({selected_year})")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            df_year['LifeExpectancy'].dropna(),
            kde=True, color='skyblue', ax=ax
        )
        ax.axvline(
            df_year['LifeExpectancy'].mean(),
            color='red', linestyle='--',
            label=f"Mean: {df_year['LifeExpectancy'].mean():.1f}"
        )
        ax.set_xlabel('Life Expectancy (years)')
        ax.set_ylabel('Number of Countries')
        ax.legend()
        st.pyplot(fig)

    with col_right:
        st.subheader(f"💰 GDP vs Life Expectancy ({selected_year})")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        scatter = ax2.scatter(
            df_year['GDP_per_capita'],
            df_year['LifeExpectancy'],
            c=df_year['Cluster'] if 'Cluster' in df_year.columns else 'blue',
            alpha=0.6, s=50, cmap='viridis'
        )
        ax2.set_xscale('log')
        ax2.set_xlabel('GDP per Capita (log scale)')
        ax2.set_ylabel('Life Expectancy (years)')
        ax2.grid(True, alpha=0.3)
        if 'Cluster' in df_year.columns:
            plt.colorbar(scatter, ax=ax2, label='Cluster')
        st.pyplot(fig2)

    st.divider()

    
    st.subheader("🇱🇰 Sri Lanka Spotlight")

    lka = df_cluster[
        (df_cluster['CountryName'] == 'Sri Lanka') &
        (df_cluster['Year'] == selected_year)
    ]

    if not lka.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "❤️ Life Expectancy",
                f"{lka['LifeExpectancy'].values[0]:.1f} years",
                delta=f"{lka['LifeExpectancy'].values[0] - avg_life:.1f} vs global avg"
            )
        with col2:
            st.metric(
                "💰 GDP per Capita",
                f"${lka['GDP_per_capita'].values[0]:,.0f}"
            )
        with col3:
            if 'Cluster_Name' in lka.columns:
                st.metric(
                    "🌍 Development Cluster",
                    lka['Cluster_Name'].values[0]
                )
    else:
        st.info("Sri Lanka data not available for selected year")


# PAGE 2 — EXPLORE INDICATORS

elif page == "📊 Explore Indicators":

    st.title("📊 Explore Development Indicators")
    st.caption("Analyze and compare indicators across countries")
    st.divider()

  
    indicators = [
        'LifeExpectancy', 'GDP_per_capita', 'InfantMortality',
        'HealthExpenditure', 'LiteracyRate', 'CO2_emissions',
        'Unemployment', 'InternetUsers'
    ]
    available = [col for col in indicators if col in df_year.columns]
    selected_indicator = st.selectbox("📌 Select Indicator", available)

    col_left, col_right = st.columns(2)

    with col_left:
    
        st.subheader(f"🏆 Top 10 Countries by {selected_indicator}")
        top10 = df_year.nlargest(10, selected_indicator)[['CountryName', selected_indicator]]
        top10 = top10.reset_index(drop=True)
        top10.index += 1
        st.dataframe(top10, use_container_width=True)

    with col_right:
    
        st.subheader(f"📉 Bottom 10 Countries by {selected_indicator}")
        bot10 = df_year.nsmallest(10, selected_indicator)[['CountryName', selected_indicator]]
        bot10 = bot10.reset_index(drop=True)
        bot10.index += 1
        st.dataframe(bot10, use_container_width=True)

    st.divider()

    
    st.subheader(f"📊 Top 10 Countries — {selected_indicator} ({selected_year})")
    fig, ax = plt.subplots(figsize=(10, 5))
    top10_plot = df_year.nlargest(10, selected_indicator)
    ax.barh(
        top10_plot['CountryName'],
        top10_plot[selected_indicator],
        color='steelblue'
    )
    ax.set_xlabel(selected_indicator)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig)

    st.divider()

    
    st.subheader("🔥 Correlation Heatmap")
    corr_cols = [col for col in available if col in df_year.columns]
    corr = df_year[corr_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr, annot=True, fmt='.2f',
        cmap='coolwarm', square=True,
        linewidths=0.5, ax=ax3
    )
    st.pyplot(fig3)

    st.divider()

   
    with st.expander("📋 View Raw Data"):
        st.dataframe(df_year, use_container_width=True)


# PAGE 3 — PREDICT LIFE EXPECTANCY

elif page == "🤖 Predict Life Expectancy":

    st.title("🤖 Predict Life Expectancy")
    st.caption("Input development indicators to predict life expectancy")
    st.divider()

    if model is None:
        st.error("❌ Model files not found. Please upload pkl files to the app folder.")
        st.stop()

    # Input sliders
    st.subheader("📌 Enter Indicators")

    col1, col2 = st.columns(2)

    with col1:
        year       = st.slider("📅 Year", 2000, 2025, 2022)
        gdp        = st.slider("💰 GDP per Capita (USD)", 500, 80000, 4000, 500)
        infant     = st.slider("👶 Infant Mortality (per 1000)", 1, 100, 20)
        health_exp = st.slider("🏥 Health Expenditure (% GDP)", 1.0, 20.0, 5.0, 0.5)

    with col2:
        literacy     = st.slider("📚 Literacy Rate (%)", 20.0, 100.0, 80.0, 0.5)
        co2          = st.slider("🌿 CO2 Emissions (tonnes)", 0.1, 30.0, 2.0, 0.1)
        unemployment = st.slider("💼 Unemployment (%)", 0.0, 40.0, 5.0, 0.5)
        internet     = st.slider("🌐 Internet Users (%)", 0.0, 100.0, 50.0, 1.0)

    st.divider()

 
    if st.button("🔮 Predict Life Expectancy", type="primary"):

       
        input_map = {
            'Year'              : year,
            'GDP_per_capita'    : gdp,
            'InfantMortality'   : infant,
            'HealthExpenditure' : health_exp,
            'LiteracyRate'      : literacy,
            'CO2_emissions'     : co2,
            'Unemployment'      : unemployment,
            'InternetUsers'     : internet,
        }

        input_values = [input_map.get(f, 0) for f in features]
        input_df     = pd.DataFrame([input_values], columns=features)
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]

      
        cluster_input  = [input_map.get(f, 0) for f in clust_feats]
        cluster_df     = pd.DataFrame([cluster_input], columns=clust_feats)
        cluster_scaled = scaler_clust.transform(cluster_df)
        cluster_id     = kmeans.predict(cluster_scaled)[0]
        cluster_label  = clust_names.get(cluster_id, f'Cluster {cluster_id}')

    
        sri_lanka_ref = 77.0

    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Prediction Complete!")
            st.metric(
                "❤️ Predicted Life Expectancy",
                f"{prediction:.1f} years",
                delta=f"{prediction - sri_lanka_ref:.1f} vs Sri Lanka"
            )
        with col2:
            st.metric("🌍 Development Cluster", cluster_label)
        with col3:
            st.metric("📅 Year", year)

        st.divider()

      
        st.subheader("📋 Your Input Summary")
        summary = pd.DataFrame({
            'Indicator' : [
                'Year', 'GDP per Capita', 'Infant Mortality',
                'Health Expenditure', 'Literacy Rate',
                'CO2 Emissions', 'Unemployment', 'Internet Users'
            ],
            'Value' : [
                year, f"${gdp:,}", f"{infant} per 1000",
                f"{health_exp}% of GDP", f"{literacy}%",
                f"{co2} tonnes", f"{unemployment}%", f"{internet}%"
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)


# PAGE 4 — COUNTRY COMPARISON

elif page == "🌍 Country Comparison":

    st.title("🌍 Country Comparison")
    st.caption("Compare two countries side by side")
    st.divider()

    all_countries = sorted(df_cluster['CountryName'].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        country1 = st.selectbox("🌍 Select Country 1", all_countries,
                                index=all_countries.index('Sri Lanka') if 'Sri Lanka' in all_countries else 0)
    with col2:
        country2 = st.selectbox("🌍 Select Country 2", all_countries,
                                index=all_countries.index('India') if 'India' in all_countries else 1)


    df_c1 = df_cluster[
        (df_cluster['CountryName'] == country1) &
        (df_cluster['Year'] == selected_year)
    ]
    df_c2 = df_cluster[
        (df_cluster['CountryName'] == country2) &
        (df_cluster['Year'] == selected_year)
    ]

    if df_c1.empty or df_c2.empty:
        st.warning("Data not available for one or both countries in selected year")
        st.stop()

   
    indicators = [
        'LifeExpectancy', 'GDP_per_capita', 'InfantMortality',
        'HealthExpenditure', 'LiteracyRate', 'Unemployment'
    ]
    available = [col for col in indicators if col in df_cluster.columns]

    st.subheader(f"📊 {country1} vs {country2} — {selected_year}")

    for ind in available:
        col_a, col_b, col_c = st.columns([2, 1, 2])
        val1 = df_c1[ind].values[0] if not df_c1.empty else 0
        val2 = df_c2[ind].values[0] if not df_c2.empty else 0

        with col_a:
            st.metric(f"{country1}", f"{val1:.2f}")
        with col_b:
            st.markdown(f"<p style='text-align:center; margin-top:20px'><b>{ind}</b></p>",
                       unsafe_allow_html=True)
        with col_c:
            st.metric(f"{country2}", f"{val2:.2f}",
                     delta=f"{val2 - val1:.2f} vs {country1}")

    st.divider()


    st.subheader(f"📈 Life Expectancy Trend — {country1} vs {country2}")

    trend1 = df_cluster[df_cluster['CountryName'] == country1].groupby('Year')['LifeExpectancy'].mean()
    trend2 = df_cluster[df_cluster['CountryName'] == country2].groupby('Year')['LifeExpectancy'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trend1.index, trend1.values, 'b-o', linewidth=2, markersize=4, label=country1)
    ax.plot(trend2.index, trend2.values, 'r-s', linewidth=2, markersize=4, label=country2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Life Expectancy (years)')
    ax.set_title(f'Life Expectancy Trend: {country1} vs {country2}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


st.divider()
st.caption("🌍 Data Source: World Bank World Development Indicators | Built with Streamlit")
