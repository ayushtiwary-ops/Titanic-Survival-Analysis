import csv
import numpy as np

# --- Step 1: Load the raw CSV ---
ages_raw = []
fares = []
survived = []

with open("train.csv", "r") as file:
    reader = csv.DictReader(file)      # DictReader treats first row as column names
    for row in reader:
        ages_raw.append(row["Age"])    # still a string at this point
        fares.append(float(row["Fare"]))
        survived.append(int(row["Survived"]))

print("Total rows loaded:", len(ages_raw))
print("First 5 ages (raw):", ages_raw[:5])

# --- Step 2: Clean ages - remove blanks, convert to float ---
ages_clean = []

for age in ages_raw:
    if age != "":               # skip empty cells
        ages_clean.append(float(age))

print("\nTotal passengers:", len(ages_raw))
print("Passengers with age recorded:", len(ages_clean))
print("Passengers with missing age:", len(ages_raw) - len(ages_clean))
# --- Step 3: NumPy statistics on age ---
ages_np = np.array(ages_clean)   # Python list → NumPy array

print("\n--- Age Statistics ---")
print("Mean age:   ", round(np.mean(ages_np), 2))
print("Median age: ", np.median(ages_np))
print("Std dev:    ", round(np.std(ages_np), 2))
print("Min age:    ", np.min(ages_np))
print("Max age:    ", np.max(ages_np))

# --- Step 4: Fare analysis using NumPy ---
fares_np = np.array(fares)

print("\n--- Fare Statistics ---")
print("Mean fare:   ", round(np.mean(fares_np), 2))
print("Median fare: ", round(np.median(fares_np), 2))

# Percentiles - what price marks the top 10% and bottom 10%?
top_10_cutoff    = np.percentile(fares_np, 90)
bottom_10_cutoff = np.percentile(fares_np, 10)

print("Top 10% fare threshold:    ", round(top_10_cutoff, 2))
print("Bottom 10% fare threshold: ", round(bottom_10_cutoff, 2))

# Boolean filtering - who is in each group?
top_10_fares    = fares_np[fares_np >= top_10_cutoff]
bottom_10_fares = fares_np[fares_np <= bottom_10_cutoff]

print("Passengers in top 10%:    ", len(top_10_fares))
print("Passengers in bottom 10%: ", len(bottom_10_fares))
fares_np >= top_10_cutoff
# produces: [False, False, True, False, True, True, ...]
# one True/False for every passenger


# keeps only the values where True appears

# --- Step 5: Survival rate by age group ---
# We need age AND survival for the same passenger
# Load both together this time

ages_with_survival = []

with open("train.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row["Age"] != "":
            ages_with_survival.append((float(row["Age"]), int(row["Survived"])))

# Split into two parallel NumPy arrays
ages_paired    = np.array([x[0] for x in ages_with_survival])
survived_paired = np.array([x[1] for x in ages_with_survival])

# Boolean masks for each age group
mask_child  = ages_paired < 15
mask_adult  = (ages_paired >= 15) & (ages_paired <= 60)
mask_senior = ages_paired > 60

# Survival rate = mean of Survived column (1s and 0s)
rate_child  = np.mean(survived_paired[mask_child])
rate_adult  = np.mean(survived_paired[mask_adult])
rate_senior = np.mean(survived_paired[mask_senior])

print("\n--- Survival Rate by Age Group ---")
print(f"Children  (< 15):    {rate_child:.1%}  ({mask_child.sum()} passengers)")
print(f"Adults    (15-60):   {rate_adult:.1%}  ({mask_adult.sum()} passengers)")
print(f"Seniors   (> 60):    {rate_senior:.1%}  ({mask_senior.sum()} passengers)")