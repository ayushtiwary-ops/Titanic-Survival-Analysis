# part4_executive.py
# Part 4 - Executive Challenge Answers

answers = """
================================================================
PART 4 — EXECUTIVE CHALLENGE
Data-Driven Survival Intelligence | Titanic Case Study
================================================================

Q1: WHO SHOULD BE PRIORITIZED FOR RESCUE TODAY?
------------------------------------------------
Based on our survival scoring model, rescue priority should follow
this order:

1. Children under 15 (57.7% historical survival — highest of any group)
2. Women regardless of class (50-97% survival vs 14-37% for men)
3. Passengers travelling with small families (IsAlone was 4th strongest predictor)
4. Elderly passengers over 60 (only 22.7% survival — most at risk)

However, applying this model directly today raises serious problems.
See Q2.

Q2: THREE ETHICAL CONCERNS IN AUTOMATED SURVIVAL PREDICTION
------------------------------------------------------------
1. BAKED-IN HISTORICAL BIAS
   The model learned from 1912 social norms where gender roles
   explicitly determined who got lifeboat access. Encoding "female"
   as a survival predictor doesn't reflect biology — it reflects
   a century-old policy of prioritizing women. Automating this
   today means encoding a historical power structure into an
   algorithm, not a neutral truth.

2. CLASS AS A PROXY FOR RACE AND WEALTH
   Pclass is the second strongest predictor. In 1912, 3rd class
   was disproportionately immigrants and non-white passengers.
   A model trained on this data would systematically score
   certain ethnic groups lower — not because of anything
   inherent but because of who could afford which ticket in 1912.
   Using this for any modern rescue or insurance decision would
   constitute illegal discrimination.

3. SMALL SAMPLE SIZE FOR SENIORS
   Only 22 passengers were over 60 in our dataset. Our 22.7%
   senior survival rate is based on just 22 people. Statistical
   decisions made from such small groups are unreliable and
   could systematically harm elderly people in real applications.

Q3: IF THIS WERE AN INSURANCE UNDERWRITING DATASET
---------------------------------------------------
The logic would flip entirely. Instead of "who needs saving,"
the question becomes "who is a financial risk to insure."

Changes I would make:

1. REMOVE GENDER as a pricing variable entirely. The EU's
   Test-Achats ruling (2012) banned gender-based insurance pricing.
   Using Sex_encoded would be illegal in most jurisdictions.

2. REFRAME PCLASS. High fare passengers (wealthy) would be LOWER
   risk because they historically had better outcomes — but insuring
   them against loss of life would mean higher payouts. The model
   logic inverts depending on what risk you're pricing.

3. ADD TEMPORAL FEATURES. A 1912 ship had no GPS, limited lifeboats,
   and no distress beacon. Modern vessels have entirely different
   survival dynamics. The training data is 114 years old — almost
   none of it transfers to modern maritime insurance.

4. FOCUS ON BEHAVIORAL RISK, not demographic risk. Modern insurance
   underwriting uses driving history, health records, lifestyle
   choices — not gender and ticket price. The Titanic model would
   be rejected by any modern actuarial review board.

================================================================
"""

print(answers)