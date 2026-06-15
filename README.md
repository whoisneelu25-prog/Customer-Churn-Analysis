# 📉 Customer Churn Analysis

Analyze subscription customer data to uncover the patterns behind cancellations. This project evaluates user activity, engagement levels, and behavioral trends to identify key churn drivers and generate prioritized retention recommendations.

Based on: [How would a Data Scientist analyze Customer Churn?](https://youtu.be/6EmjRXUcARc)

---

## 📁 Project Structure

```
churn-analysis/
│
├── data/
│   ├── raw/
│   │   └── customers_raw.csv          # Raw customer dataset (50 records)
│   └── processed/
│       └── customers_enriched.csv     # Feature-engineered output
│
├── notebooks/
│   └── churn_analysis_walkthrough.ipynb  # Step-by-step analysis notebook
│
├── scripts/
│   └── churn_analysis.py              # Full analysis pipeline (CLI)
│
├── outputs/
│   ├── customers_enriched.csv         # Dataset with derived features
│   ├── at_risk_customers.csv          # Flagged at-risk active customers
│   ├── churn_drivers.csv              # Feature correlation table
│   ├── recommendations.json           # Retention actions (JSON)
│   ├── summary_metrics.json           # Headline metrics (JSON)
│   ├── engagement_comparison.png      # Chart: churned vs retained engagement
│   ├── churn_by_segment.png           # Chart: churn by plan/contract/tenure
│   ├── churn_drivers.png              # Chart: feature correlations
│   └── cohort_retention.png           # Chart: retention by tenure cohort
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚨 What the Raw Data Contains

| Column | Description |
|---|---|
| `customer_id` | Unique identifier |
| `age` | Customer age |
| `gender` | Male / Female |
| `tenure_months` | Months as a subscriber |
| `plan` | Basic / Standard / Premium |
| `monthly_charge` | Monthly subscription fee |
| `total_charges` | Cumulative revenue from customer |
| `contract_type` | Month-to-month / One year / Two year |
| `payment_method` | Credit card / Bank transfer / Mailed check |
| `num_products` | Number of products/features used |
| `support_calls` | Support contacts in past 90 days |
| `login_frequency` | Logins in past 30 days |
| `last_active_days` | Days since last login |
| `nps_score` | Net Promoter Score (1–10) |
| `churned` | 1 = cancelled, 0 = active |

---

## 🔍 Analysis Steps

| Step | What it does |
|---|---|
| 1 | Load & validate raw customer data |
| 2 | Engineer features: engagement score, LTV, at-risk flag, tenure bands |
| 3 | Compute headline churn metrics (rate, MRR at risk, avg tenure) |
| 4 | Segment analysis: churn by plan, contract type, tenure cohort |
| 5 | Rank churn drivers by correlation strength |
| 6 | Cohort retention table by tenure band |
| 7 | Generate prioritized retention recommendations |
| 8 | Export enriched dataset, at-risk list, and charts |

---

## 📊 Key Findings (Sample Dataset)

| Finding | Insight |
|---|---|
| **Short tenure = high churn** | Most cancellations happen in months 1–6 |
| **Low engagement → churn** | Churned users log in ~1–2x/month vs 18–25x for retained |
| **High support calls** | Churned customers average 8+ support calls vs 1 for retained |
| **Month-to-month contracts** | Churn rate 3–5× higher than annual/biennial contracts |
| **NPS detractors** | Churned users score avg 1–2 vs 8–10 for loyal customers |
| **Basic plan over-represented** | Basic plan customers churn at the highest rate |

---

## 💡 Top Retention Recommendations

| Priority | Segment | Action |
|---|---|---|
| 🔴 HIGH | New customers (0–6 mo) | 90-day onboarding journey with check-in at day 30 |
| 🔴 HIGH | Low-engagement users | Re-engagement email when inactive for 14+ days |
| 🔴 HIGH | High-friction users | Proactive CSM outreach after 3+ support calls |
| 🟡 MEDIUM | NPS Detractors | Route score ≤6 customers to retention specialist in 48h |
| 🟡 MEDIUM | Month-to-month | Offer 15–20% discount to upgrade to annual plan |
| 🟢 LOW | Low product adoption | In-app feature nudges + adoption tracking as KPI |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/churn-analysis.git
cd churn-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
# With built-in sample data
python scripts/churn_analysis.py

# With your own CSV
python scripts/churn_analysis.py --input data/raw/customers_raw.csv
```

### 4. Open the notebook
```bash
jupyter notebook notebooks/churn_analysis_walkthrough.ipynb
```

---

## 🛠 Tech Stack

- **Python 3.11+**
- **pandas** — data manipulation & segmentation
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualizations
- **scikit-learn** — (extensible for ML churn prediction)
- **Jupyter** — interactive analysis

---

## 📈 Extending This Project

Ideas to build on top of this foundation:

- [ ] Add a **logistic regression** or **XGBoost** churn prediction model
- [ ] Build a **real-time at-risk dashboard** in Streamlit or Tableau
- [ ] Connect to live CRM data (Salesforce, HubSpot)
- [ ] Add **A/B test tracking** to measure retention intervention impact
- [ ] Schedule weekly churn reports via email

---

## 📚 Reference

Tutorial: [How would a Data Scientist analyze Customer Churn? – YouTube](https://youtu.be/6EmjRXUcARc)
