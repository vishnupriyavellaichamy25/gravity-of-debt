import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Phase 1: EDA & Feature Engineering\n\nThis notebook covers the exploratory data analysis of the LendingClub dataset, missing value handling, feature distributions, class imbalance checks, and correlation analysis."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n%matplotlib inline"),
    nbf.v4.new_code_cell("df = pd.read_csv('../data/lending_club_sample.csv')\ndf.head()"),
    nbf.v4.new_markdown_cell("## Missing Value Heatmap"),
    nbf.v4.new_code_cell("plt.figure(figsize=(10,6))\nsns.heatmap(df.isnull(), cbar=False, cmap='viridis')\nplt.title('Missing Values Heatmap')\nplt.show()"),
    nbf.v4.new_markdown_cell("## Class Imbalance Check"),
    nbf.v4.new_code_cell("plt.figure(figsize=(6,4))\nsns.countplot(data=df, x='loan_status')\nplt.title('Target Variable Distribution')\nplt.show()\n\nprint(df['loan_status'].value_counts(normalize=True))"),
    nbf.v4.new_markdown_cell("## Distributions & Correlations"),
    nbf.v4.new_code_cell("plt.figure(figsize=(10,6))\nsns.histplot(df['loan_amnt'], bins=50, kde=True)\nplt.title('Loan Amount Distribution')\nplt.show()"),
    nbf.v4.new_code_cell("numeric_cols = df.select_dtypes(include=[np.number]).columns\ncorr = df[numeric_cols].corr()\n\nplt.figure(figsize=(12,10))\nsns.heatmap(corr, cmap='coolwarm', annot=False)\nplt.title('Correlation Matrix')\nplt.show()"),
    nbf.v4.new_markdown_cell("## Feature Engineering Overview\n\nSee `src/preprocess.py` for the complete pipeline including SMOTE and scaling.")
]

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/01_eda_and_modeling.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
