# Retail & Warehouse Sales Intelligence

## Project Overview
An AI-assisted data analytics dashboard for understanding retail sales, warehouse sales, retail transfers, product performance, supplier concentration, and unusual operational records.

## Dataset
**Source:** Kaggle — Retail Sales Data with Seasonal Trends & Marketing  
**Dataset page:** https://www.kaggle.com/dsv/9428891

The supplied CSV contains 30,000 records and 9 columns:
- YEAR
- MONTH
- SUPPLIER
- ITEM CODE
- ITEM DESCRIPTION
- ITEM TYPE
- RETAIL SALES
- RETAIL TRANSFERS
- WAREHOUSE SALES

## Objectives
1. Measure retail, warehouse, and transfer activity.
2. Identify high-performing products, item types, and suppliers.
3. Compare retail activity with warehouse activity.
4. Identify concentration and operational patterns.
5. Use machine learning to flag unusual records for investigation.
6. Convert analytical findings into practical business recommendations.

## AI Component
The dashboard uses **Isolation Forest** from scikit-learn for unsupervised anomaly detection. It flags records with unusual combinations of retail sales, retail transfers, and warehouse sales.

An anomaly is an **investigation candidate**, not proof of an error, fraud, or misconduct.

## Important Dataset Limitation
The supplied dataset contains records from **2020** and only the months **January, March, July, and September**. Therefore, the project intentionally does not claim a complete yearly seasonal pattern or produce a conventional long-horizon time-series forecast.

## How to Run
1. Install Python 3.10+.
2. Put these files in the same folder:
   - `retail_warehouse_ai.py`
   - `Retail and wherehouse Sale.csv`
   - `requirements.txt`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the dashboard:
   ```bash
   streamlit run retail_warehouse_ai.py
   ```

## Output
The Streamlit dashboard provides:
- Executive KPI cards
- Monthly activity analysis
- Item-type analysis
- Top product analysis
- Supplier performance
- AI anomaly detection
- Automated business recommendations

## Tech Stack
Python, Pandas, NumPy, Plotly, Streamlit, Scikit-learn.

## Project Type
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 project.
