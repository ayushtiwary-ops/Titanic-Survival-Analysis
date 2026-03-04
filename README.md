# Titanic-Survival-Analysis
This is a full data analytics project, not just a script. It's the IIT Gandhinagar capstone bonus assignment, using the Titanic dataset. Here's what makes it fundamentally different from everything you've done so far:

What You're Actually Building
Four interconnected deliverables submitted together:

A cleaned dataset (CSV file)
A Python script/notebook with all analysis code
A 3-page PDF insight report — written like a consultant, not a student
A manual survival scorer built from scratch using only NumPy (no sklearn, no ML libraries)


Part 1 — Raw Data Exploration (Lists + NumPy Only)
Skill being tested: Can you work with raw data before Pandas? This is deliberately restrictive to test fundamentals.
You load the CSV with Python's built-in csv module, pull out columns into Python lists, then use NumPy to compute statistics. The key tasks:

Pull the Age column as a Python list, remove NaN values manually
np.mean(), np.median(), np.std() on the cleaned list
Create a NumPy array of Fare, then use np.percentile() to find top 10% and bottom 10%
Use NumPy boolean filtering to compare survival rates across 3 age brackets: under 15, 15–60, over 60

The insight question — "Is age linearly related to survival?" — expects you to show the actual survival rates per bracket and note that children had higher survival (lifeboats), elderly had the lowest, but the middle group was roughly flat. That's not linear — it's more of a U-shape. One line of statistical justification wins full marks here.

Part 2 — Advanced Pandas Engineering
This is the meatiest section — 40%+ of the work. Six tasks:
Missing value handling — fill Age with the median within each passenger class, not a global median. This is the correct data science approach because 1st class passengers were older on average.
Feature engineering — three new columns you create:

FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0
FarePerPerson = Fare / FamilySize

Categorical binning — pd.cut() for AgeGroup and FareGroup. The logic matters more than the labels.
Pivot tables — pd.pivot_table() for survival by Gender+Class and by FareGroup+Embarked. These should show the famous "women and children first" signal clearly.
Correlation matrix — df.corr() on numerical columns only. This feeds directly into Part 3.
Insight question — "Does wealth dominate gender in predicting survival?" The actual answer from the data: No — gender dominates. Women in 3rd class survived at higher rates than men in 1st class. Wealth helped, but gender was the primary signal. You need to support this with the grouped stats.

Part 3 — Manual Survival Scorer (The Hard Part)
No sklearn. No ML. Just NumPy arithmetic — this is what separates this assignment from a standard data science course.
You build a weighted scoring formula:
SurvivalScore = w1×Gender + w2×Pclass + w3×AgeGroup + w4×FareGroup + w5×FamilySize
The process:

Normalize each feature using (x - min) / (max - min) with NumPy
Assign weights based on what the correlation matrix showed you in Part 2
Sigmoid the score or just threshold it at 0.5 to classify as survived/not
Manually compute accuracy, precision, recall, and confusion matrix using NumPy boolean operations — no sklearn.metrics allowed

The insight question asks if your handcrafted score beats random guessing. A random classifier on this dataset gets ~38% accuracy (baseline survival rate). Your weighted scorer should comfortably beat 70%. If it doesn't, your weights are wrong.

Part 4 — Executive Challenge (The Writing Part)
Three written questions that test critical thinking, not coding:
Q1 — Who should be prioritized today? Your model would say: children first, then women, then elderly. But ethically this has problems — see Q2.
Q2 — 3 ethical concerns — the evaluator wants you to think beyond the code:

Algorithmic bias baked in from 1912 social norms (gender roles, class hierarchy)
Using demographic features (gender, class) in automated decisions is legally problematic in modern contexts
Historical survival patterns reflecting privilege, not need

Q3 — Insurance underwriting — you'd flip the logic. Instead of "who needs rescuing," you'd ask "who is a risk." High FarePerPerson (wealthy) would be lower risk. But using gender as a pricing variable is now illegal in many countries (EU banned it in insurance in 2012). This is the kind of nuanced answer IIT evaluators love.

Bonus Challenge (Very Difficult)
All three tasks require advanced NumPy:

Recreate groupby operations using only np.where(), np.unique(), and boolean masks — no Pandas
Vectorize everything — no Python loops anywhere in the computation
Build a predict_survival(passenger_dict) function that takes a single passenger as input and returns a 0 or 1

This is genuinely hard. The vectorized groupby without Pandas requires understanding how to use np.unique() with return_inverse=True to manually recreate what groupby does.

What Makes This Assignment Different from All Previous Ones
DimensionDay 7–8 AssignmentsThis AssignmentScopeSingle script, 1–2 hoursMulti-file project, 6–8 hoursDataNo real dataReal 891-row datasetOutputConsole outputPDF report + cleaned CSV + codeThinking requiredApply syntaxJustify statistical choicesEvaluation styleRubric checklistConsultant-level judgment

Recommended Execution Order

Download Titanic dataset from Kaggle first
Part 1 — raw NumPy work (~45 min)
Part 2 — Pandas engineering (~60 min)
Part 3 — scorer build + manual metrics (~45 min)
Part 4 — write the executive answers (~30 min)
