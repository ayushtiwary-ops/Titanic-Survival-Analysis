import pandas as pd
import numpy as np

# --- Load the data ---
df = pd.read_csv("train.csv")

print("Shape:", df.shape)          # (rows, columns)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# --- Step 1: Fill missing ages with median per class ---
print("\nMedian age per class BEFORE filling:")
print(df.groupby("Pclass")["Age"].median())

df["Age"] = df.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median())
)

print("\nMissing ages after filling:", df["Age"].isnull().sum())

# --- Step 2: Fill missing Embarked with mode ---
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("Missing Embarked after filling:", df["Embarked"].isnull().sum())

# --- Step 3: Feature Engineering ---
# FamilySize: how many people travelling together including themselves
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# IsAlone: 1 if travelling solo, 0 if with family
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# FarePerPerson: split fare equally across family
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

print("\nNew features sample:")
print(df[["Name", "FamilySize", "IsAlone", "FarePerPerson"]].head(8))
# --- Step 4: Categorical bins ---
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 15, 60, 100],
    labels=["Child", "Adult", "Senior"]
)

df["FareGroup"] = pd.cut(
    df["Fare"],
    bins=[0, 15, 77.96, 600],
    labels=["Low", "Medium", "High"]
)

print("\nAgeGroup distribution:")
print(df["AgeGroup"].value_counts())
print("\nFareGroup distribution:")
print(df["FareGroup"].value_counts())

# --- Step 4: Categorical bins ---
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 15, 60, 100],
    labels=["Child", "Adult", "Senior"]
)

df["FareGroup"] = pd.cut(
    df["Fare"],
    bins=[0, 15, 77.96, 600],
    labels=["Low", "Medium", "High"]
)

print("\nAgeGroup distribution:")
print(df["AgeGroup"].value_counts())
print("\nFareGroup distribution:")
print(df["FareGroup"].value_counts())          
# --- Step 5: Pivot Tables ---
print("\n=== Survival by Gender and Class ===")
pivot1 = pd.pivot_table(
    df,
    values="Survived",
    index="Sex",
    columns="Pclass",
    aggfunc="mean"
)
print(pivot1.round(2))

print("\n=== Survival by FareGroup and Embarked ===")
pivot2 = pd.pivot_table(
    df,
    values="Survived",
    index="FareGroup",
    columns="Embarked",
    aggfunc="mean"
)
print(pivot2.round(2))

# --- Step 6: Correlation Matrix ---
print("\n=== Correlation with Survival ===")
numeric_cols = ["Survived", "Pclass", "Age", "SibSp",
                "Parch", "Fare", "FamilySize", "FarePerPerson"]
corr = df[numeric_cols].corr()
print(corr["Survived"].sort_values(ascending=False).round(3))

# --- Step 7: Encode Sex for correlation ---
df["Sex_encoded"] = (df["Sex"] == "female").astype(int)
corr2 = df[["Survived", "Sex_encoded", "Pclass", "Fare",
             "FamilySize", "Age", "IsAlone"]].corr()

print("\n=== Top features by correlation strength ===")
feature_corr = corr2["Survived"].drop("Survived").abs().sort_values(ascending=False)
print(feature_corr.round(3))     

# Add to part2_pandas.py at the very end
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved: titanic_cleaned.csv")