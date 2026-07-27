import pandas as pd

# Load CSV file
df = pd.read_csv("45849-s3-sf_TEXT.csv")

# Create working dataset
data = df.copy()

# Check data
print(data.head())
print(data.shape)
print(data.columns.tolist())

# Check age
print("Missing age:")
print(data["Age"].isna().sum())

print("\nAge distribution:")
print(data["Age"].describe())

# Check founder responses
print("\nFounder responses:")
print(data["Founder"].value_counts(dropna=False))

# Check founder status
print("\nFounder status:")
print(data["Founder_Status"].value_counts(dropna=False))


# -----------------------------
# Apply exclusion criteria
# -----------------------------

print("Starting sample:", len(data))

# 1. Remove missing age
data = data[data["Age"].notna()]
print("After missing age removed:", len(data))

# 2. Remove participants under 18
data = data[data["Age"] >= 18]
print("After under 18 removed:", len(data))

# 3. Remove missing founder response
data = data[data["Founder"].notna()]
print("After missing founder response removed:", len(data))

# 4. Remove non-founders
data = data[data["Founder"] != "No"]
print("After non-founders removed:", len(data))


# -----------------------------
# Check questionnaire completion
# -----------------------------

# CPM items
cpm_items = [f"CPM_{i}" for i in range(1, 28)]

# PtP items
ptp_items = [f"PtP_{i}" for i in range(1, 65)]

# Count completed items for each participant
data["CPM_completed"] = data[cpm_items].notna().sum(axis=1)
data["PtP_completed"] = data[ptp_items].notna().sum(axis=1)

# Display completion counts
print("\nQuestionnaire completion:")
print(data[["CPM_completed", "PtP_completed"]])
