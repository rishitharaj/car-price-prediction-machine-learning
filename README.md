# 🚗 Car Price Prediction – Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

A Chinese automobile company plans to enter the **US automobile market** by setting up a local manufacturing unit. Since pricing dynamics in the US differ significantly from the Chinese market, the company wants to understand **which factors influence car prices** and **how well those factors explain price variations**.

This project applies **machine learning regression models** to analyze car pricing data from the US market and identify the most influential features affecting car prices.

---

## 🎯 Business Objective

- Identify key variables affecting car prices in the US market  
- Build predictive models to estimate car prices accurately  
- Compare multiple regression models and select the best performer  
- Provide actionable insights to support pricing strategy and vehicle design decisions  

---

## 📂 Dataset Description

The dataset contains technical and categorical details of cars sold in the US market, including:

- Engine specifications  
- Vehicle dimensions  
- Performance metrics  
- Fuel type and aspiration  
- Brand and categorical attributes  

**Target Variable:** `price`  
**Independent Variables:** All other features  

---

## ⚙️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook/Google Colab  

---

## 🔧 Data Preprocessing

- Loaded dataset using Pandas  
- Handled missing values (if any)
- Applied **one-hot encoding** for categorical variables  
- Split data into **training (80%) and testing (20%)** sets  
- Applied **feature scaling** where required (Linear Regression & SVR)  

---

## 🤖 Models Implemented

The following regression models were trained and evaluated:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Support Vector Regressor (SVR)  

---

## 📊 Model Evaluation

Models were evaluated using:

- **R² Score**
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

---

## 🔍 Performance Comparison

| Model | R² Score | MSE | MAE |
|------|---------|---------|---------|
| Linear Regression | -1.237 | 1.77 × 10⁸ | 7645 |
| Decision Tree | 0.867 | 1.05 × 10⁷ | 2098 |
| Random Forest | 0.953 | 3.67 × 10⁶ | 1365 |
| Gradient Boosting | 0.932 | 5.40 × 10⁶ | 1686 |
| SVR | -0.102 | 8.70 × 10⁷ | 5705 |

---

## 🏆 Best Model: Random Forest Regressor

The **Random Forest Regressor** delivered the best overall performance:

- **Highest R² Score:** 0.953  
- **Lowest prediction error:** MAE ≈ **$1,365**  
- Effectively captures **non-linear relationships**  
- Robust against overfitting due to **ensemble learning**  

---

## 🔍 Feature Importance Analysis

Key features influencing car prices include:

- Engine Size  
- Curb Weight
- highwaympg
- Horsepower   
- Car Width  

These factors play a critical role in determining manufacturing cost, performance, and perceived value in the US automobile market.

---

## 🔧 Hyperparameter Tuning

**GridSearchCV** was applied to the Random Forest model to improve performance.

### 📈 Performance Improvement

| Model | R² Score | MSE | MAE |
|------|---------|---------|---------|
| Random Forest (Before Tuning) | 0.9535 | 3,674,318 | 1365.35 |
| **Random Forest (After Tuning)** | **0.9556** | **3,501,314** | **1333.30** |

✔ Hyperparameter tuning improved generalization and reduced prediction error.

---

## 💡 Business Insights & Recommendations

- Car pricing in the US is strongly influenced by **engine performance and vehicle dimensions**  
- Manufacturers can design vehicles for **specific price segments** by optimizing these features  
- **Ensemble models** are recommended for pricing strategies due to their reliability  
- The tuned Random Forest model is suitable for **real-world deployment**  

---

## ✅ Conclusion

This project demonstrates the effectiveness of machine learning in understanding and predicting car prices in a new market. The **tuned Random Forest Regressor** explained approximately **96% of the variance** in car prices, making it the most reliable model for this problem. The insights derived can support strategic decision-making for market entry and competitive positioning.

---

## 📁 Repository Structure

Car Price Prediction Machine Learning
├── Car_Price_Prediction.ipynb
├── data/
│   └── Car Price Assignment.csv
├── README.md


**👤 Author**

Rishitha Raj

Machine Learning & Data Science Enthusiast
