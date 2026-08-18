# 🌍 World Development Indicators — Life Expectancy Prediction & Clustering

A Data Science project analyzing global development patterns and predicting life expectancy using World Bank data.

---

## 📌 Overview

This project applies machine learning to the **World Bank World Development Indicators (WDI)** dataset to:
- **Predict life expectancy** using development indicators — *Regression*
- **Group countries** by development profile — *Clustering*
- **Analyze Sri Lanka's position** against global and regional peers

Completed as part of **COSC 44343 / BECS 44613 — Data Science**
**University of Kelaniya** | Academic Year 2024/2025

---

## 📊 Dataset

| Detail | Info |
|---|---|
| Source | [World Bank WDI — Kaggle](https://www.kaggle.com/datasets/theworldbank/world-development-indicators) |
| Coverage | 217 countries, 2000–2025 |
| Raw size | ~5.6 million rows |
| After preprocessing | ~4,000 rows, 9 features |

---

## 🔧 Techniques Used

| Technique | Purpose |
|---|---|
| Regression (XGBoost) | Predict life expectancy |
| K-Means Clustering | Group countries by development level |
| PCA | Visualize clusters in 2D |
| GridSearchCV | Hyperparameter tuning |

---

## 📈 Results

| Model | R² Score |
|---|---|
| Linear Regression | ~0.82 |
| Random Forest | ~0.94 |
| **XGBoost (Best)** | **~0.96** |

- **Top feature:** Infant Mortality Rate
- 🇱🇰 **Sri Lanka** → Middle-Income Developing cluster, above cluster average in life expectancy

---

## 🛠️ Technologies

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Google Colab

---

## 🚀 Web App

Interactive Streamlit app — *Coming soon*

---


