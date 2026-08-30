# 🌍 Predicting Life Expectancy Using World Development Indicators

A machine learning project that predicts country-level life expectancy and 
identifies global development clusters using World Bank data (2000–2025).

---

## 🎯 What This Project Does

Life expectancy varies by 30 years between the world's richest and poorest nations.
This project uses data science to answer:
- Which development factors most influence how long people live?
- Can we accurately predict a country's life expectancy?
- Where does Sri Lanka stand compared to global and regional peers?

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [World Bank World Development Indicators](https://www.kaggle.com/datasets/theworldbank/world-development-indicators) |
| **Coverage** | 217 countries, 2000–2025 |
| **Target variable** | Life Expectancy at Birth |
| **Key features** | GDP, Infant Mortality, Health Expenditure, Literacy Rate, CO2 Emissions, Unemployment, Internet Usage |

---

## 📈 Results

### Regression — Life Expectancy Prediction

| Model | R² | RMSE |
|---|---|---|
| Linear Regression | 0.8777 | 3.21 yrs |
| Random Forest | 0.9691 | 1.61 yrs |
| **XGBoost (Best)** | **0.9752** | **1.56 yrs** |

**#1 most important feature:** Infant Mortality Rate

### Clustering — Country Development Groups

| Cluster | Countries |
|---|---|
| Advanced Economies | USA, Germany, Japan |
| High-Income Developing | China, Brazil, Malaysia |
| Middle-Income Developing | 🇱🇰 Sri Lanka, India, Philippines |
| Low-Income Developing | Nigeria, Ethiopia, Afghanistan |

### 🇱🇰 Sri Lanka

Sri Lanka ranks in the **Middle-Income Developing** cluster yet achieves a life 
expectancy **+6.9 years above** the South Asian average — driven by strong 
literacy rates and healthcare investment relative to GDP.

---

## 🖥️ Dashboard

👉 **[Live App](https://your-app-link.streamlit.app)**

Four pages:
- **Overview** — global snapshot and Sri Lanka spotlight
- **Explore** — indicator comparisons and correlation heatmap
- **Predict** — enter any country's indicators and get a life expectancy prediction
- **Compare** — side-by-side country comparison with trend lines

---

## 🛠️ Tech Stack

Python · Pandas · Scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit · Google Colab

---

## 👥 Team

- Senuri De Silva
- Sanduni Jayasinghe
- Janani Gunathilaka

**Department of Statistics and Computer Science
University of Kelaniya | 2024/2025**

---

## 📚 References

- World Bank. (2025). *World Development Indicators*. https://data.worldbank.org
- Chen & Guestrin. (2016). *XGBoost: A Scalable Tree Boosting System*
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*
