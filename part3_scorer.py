import pandas as pd
import numpy as np

# --- Load and prepare data (same as Part 2) ---
df = pd.read_csv("train.csv")

# Fill missing values
df["Age"] = df.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median())
)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Feature engineering
df["FamilySize"]   = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]      = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
df["Sex_encoded"]  = (df["Sex"] == "female").astype(int)

# --- Step 1: Normalize features to 0-1 range using NumPy ---
# Why normalize? Features are on different scales.
# Age goes 0-80, Fare goes 0-500, Sex is 0 or 1.
# Without normalization, Fare would dominate just because
# its numbers are bigger. Normalizing puts everyone on equal footing.

def normalize(arr):
    """Min-max normalization: shifts everything to 0-1 range."""
    return (arr - arr.min()) / (arr.max() - arr.min())

sex_norm    = normalize(df["Sex_encoded"].values)
pclass_norm = normalize((4 - df["Pclass"]).values)  # flip: 1st class → high score
fare_norm   = normalize(df["FarePerPerson"].values)
alone_norm  = normalize((1 - df["IsAlone"]).values)  # flip: not alone → high score
age_norm    = normalize((80 - df["Age"]).values)     # flip: younger → higher score

print("Normalization check (should all be 0.0 to 1.0):")
for name, arr in [("sex", sex_norm), ("pclass", pclass_norm),
                   ("fare", fare_norm), ("alone", alone_norm), ("age", age_norm)]:
    print(f"  {name}: min={arr.min():.2f} max={arr.max():.2f}") 

# --- Step 2: Apply weights and compute survival score ---
# Weights based on correlation strength we found in Part 2
w_sex    = 0.543
w_pclass = 0.338
w_fare   = 0.257
w_alone  = 0.203
w_age    = 0.047

# Weighted sum — each feature contributes proportionally
total_weight = w_sex + w_pclass + w_fare + w_alone + w_age

survival_score = (
    w_sex    * sex_norm +
    w_pclass * pclass_norm +
    w_fare   * fare_norm +
    w_alone  * alone_norm +
    w_age    * age_norm
) / total_weight

print("\nSurvival score sample (first 10):")
for i in range(10):
    print(f"  Passenger {i+1}: score={survival_score[i]:.3f}  actual={df['Survived'].values[i]}")

# --- Step 3: Classify using threshold 0.5 ---
predictions = (survival_score >= 0.5).astype(int)

print("\nPrediction distribution:")
print(f"  Predicted survived: {predictions.sum()}")
print(f"  Predicted died:     {(predictions == 0).sum()}")
print(f"  Actual survived:    {df['Survived'].sum()}")
print(f"  Actual died:        {(df['Survived'] == 0).sum()}")

# --- Step 4: Manual metrics using only NumPy ---
actual = df["Survived"].values

# Confusion matrix components
TP = int(np.sum((predictions == 1) & (actual == 1)))  # predicted survived, actually survived
TN = int(np.sum((predictions == 0) & (actual == 0)))  # predicted died, actually died
FP = int(np.sum((predictions == 1) & (actual == 0)))  # predicted survived, actually died
FN = int(np.sum((predictions == 0) & (actual == 1)))  # predicted died, actually survived

print("\n=== Confusion Matrix ===")
print(f"                 Predicted Survived  Predicted Died")
print(f"Actually Survived       {TP}                {FN}")
print(f"Actually Died           {FP}                {TN}")

# Metrics
accuracy  = (TP + TN) / len(actual)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n=== Performance Metrics ===")
print(f"  Accuracy  : {accuracy:.1%}  (overall correct predictions)")
print(f"  Precision : {precision:.1%}  (of predicted survivors, how many actually survived)")
print(f"  Recall    : {recall:.1%}  (of actual survivors, how many did we catch)")
print(f"  F1 Score  : {f1:.1%}  (balance of precision and recall)")

# Baseline comparison
baseline = max(df["Survived"].mean(), 1 - df["Survived"].mean())
print(f"\n  Random guess baseline : {baseline:.1%}")
print(f"  Our model             : {accuracy:.1%}")
print(f"  Improvement over random: +{(accuracy - baseline):.1%}")

# --- Step 5: predict_survival function (Bonus requirement) ---
def predict_survival(passenger_dict):
    """
    Predict survival for a single passenger.
    Input: dictionary with keys: Sex, Pclass, Fare, SibSp, Parch, Age
    Output: (prediction, score) where prediction is 0 or 1
    """
    sex    = 1 if passenger_dict["Sex"] == "female" else 0
    pclass = passenger_dict["Pclass"]
    fare   = passenger_dict["Fare"]
    family = passenger_dict["SibSp"] + passenger_dict["Parch"] + 1
    age    = passenger_dict["Age"]
    alone  = 1 if family == 1 else 0
    fare_per_person = fare / family

    # Normalize using training data ranges (approximate)
    sex_n    = sex
    pclass_n = (4 - pclass) / 2
    fare_n   = min(fare_per_person / 150, 1.0)
    alone_n  = 1 - alone
    age_n    = (80 - age) / 80

    score = (
        w_sex * sex_n +
        w_pclass * pclass_n +
        w_fare * fare_n +
        w_alone * alone_n +
        w_age * age_n
    ) / total_weight

    return int(score >= 0.5), round(score, 3)

# Test it on three very different passengers
test_cases = [
    {"Sex": "female", "Pclass": 1, "Fare": 200, "SibSp": 1, "Parch": 0, "Age": 35},
    {"Sex": "male",   "Pclass": 3, "Fare": 8,   "SibSp": 0, "Parch": 0, "Age": 25},
    {"Sex": "female", "Pclass": 3, "Fare": 12,  "SibSp": 0, "Parch": 2, "Age": 10},
]

print("\n=== predict_survival() test ===")
for p in test_cases:
    pred, score = predict_survival(p)
    result = "SURVIVED" if pred == 1 else "DIED"
    print(f"  {p['Sex']:6} | Class {p['Pclass']} | Age {p['Age']:2} | "
          f"Fare £{p['Fare']:3} → score={score} → {result}")