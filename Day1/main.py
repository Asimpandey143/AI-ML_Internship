from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (make sure 'titanic.csv' is in your folder)
df = pd.read_csv('Titanic-Dataset.csv')

# See the first 5 rows
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())

# Fill missing ages with the median age
#df['Age'].fillna(df['Age'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())


# Fill missing Embarked values with the most common value
df['Embarked'].fillna(df['Embarked'].mode()[0])

#Convert "Sex" from "male"/"female" to 0/1.
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

#For columns with more than two categories (like "Embarked"), use one-hot encoding:
df = pd.get_dummies(df, columns=['Embarked'])

#Step 5: Normalize/Standardize Numerical Features
#Standardization rescales numbers so that they have a mean of 0 and a standard deviation of 1.
#Normalization (optional) rescales values to a 0-1 range.

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Function to remove outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Remove outliers from Age and Fare
df = remove_outliers_iqr(df, 'Age')
df = remove_outliers_iqr(df, 'Fare')

#Visualize and Remove Outliers
# --- Set Style ---
sns.set(style="whitegrid")
plt.figure(figsize=(18, 12))

# ---------- PLOT 1: Countplot - Sex vs Survived ----------
plt.subplot(2, 3, 1)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Sex")
plt.xticks([0, 1], ['Male', 'Female'])

# ---------- PLOT 2: Histogram - Age distribution by Survived ----------
plt.subplot(2, 3, 2)
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
plt.title("Age Distribution by Survival")

# ---------- PLOT 3: Boxplot - Fare vs Pclass ----------
plt.subplot(2, 3, 3)
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df)
plt.title("Fare by Passenger Class and Survival")

# --- Layout fix ---
plt.tight_layout()
plt.suptitle("Titanic Dataset - Combined EDA Plots", fontsize=16, y=1.02)
plt.show()
