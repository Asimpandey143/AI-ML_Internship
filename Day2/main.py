import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

# Save summary statistics (mean, median, std, etc.)
summary_stats = df.describe(include='all')
summary_stats.to_csv("summary_statistics.csv")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

n = len(numeric_cols)
rows = n
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * n))
for idx, col in enumerate(numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[idx][0])
    axes[idx][0].set_title(f"Histogram of {col}")

    sns.boxplot(x=df[col], ax=axes[idx][1])
    axes[idx][1].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.savefig("eda_hist_box_combined.png")
plt.close()

corr = df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_correlation.png")
plt.close()

pairplot_cols = ['Age', 'Fare', 'Pclass', 'Survived']
sns.pairplot(df[pairplot_cols].dropna(), hue='Survived')
plt.savefig("eda_pairplot.png")
plt.close()

# Combine all three into one single canvas
from PIL import Image

img1 = Image.open("eda_hist_box_combined.png")
img2 = Image.open("eda_correlation.png")
img3 = Image.open("eda_pairplot.png")

width = max(img1.width, img2.width, img3.width)
height = img1.height + img2.height + img3.height

final_img = Image.new("RGB", (width, height), "white")
final_img.paste(img1, (0, 0))
final_img.paste(img2, (0, img1.height))
final_img.paste(img3, (0, img1.height + img2.height))
final_img.save("eda_summary.png")
