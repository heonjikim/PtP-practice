# Load CSV file
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import skew
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import pingouin as pg
from pyampute.exploration.mcar_statistical_tests import MCARTest
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

df = pd.read_csv("45849-s3-sf_TEXT.csv")

# Remove question-text row and ImportId row
df = df.iloc[2:].copy()

# Create working dataset
data = df.copy()

print("Starting sample:", len(data))

# Check variables
print(data.columns.tolist())

#Remove irrelevant variables
data = data.drop(columns=[
    'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)', 'RecordedDate', 'ResponseId', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail', 
    'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage', 'Q_RecaptchaScore', 'Gender_4_TEXT', 'Founder Stage', 'Projects', 
    'Uncertainty', 'Mini-IPIP 1', 'Mini-IPIP 2', 'Mini-IPIP 3', 'Mini-IPIP 4', 'IPIP 120 1', 'IPIP 120 2', 'IPIP 120 3', 'IPIP 120 4', 'APS 1', 'APS 2', 'APS 3', 'APS 4', 
    'APS 5', 'APS 6', 'APS 7', 'APS 8', 'APS 9', 'APS 10', 'APS 11', 'APS 12', 'APS 13', 'APS 14', 'APS 15', 'APS 16', 'APS 17', 'APS 18', 'APS 19', 'APS 20', 'clicked', 
    'norms', 'project', 'source', 'results', 'Q_DataPolicyViolations', 'CPM 2', 'CPM 3', 'CPM 5', 'CPM 6', 'CPM 8', 'CPM 9', 'CPM 11', 'CPM 12', 'CPM 14', 'CPM 15', 'CPM 17', 
    'CPM 18', 'CPM 20', 'CPM 21', 'CPM 23', 'CPM 24', 'CPM 26', 'CPM 27'
])
print(data.columns.tolist())

# CPM questionnaire items (change text to numbers)
cpm_items = [
    "CPM 1",
    "CPM 4",
    "CPM 7",
    "CPM 10",
    "CPM 13",
    "CPM 16",
    "CPM 19",
    "CPM 22",
    "CPM 25"
]
cpm_mapping = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5
}
data[cpm_items] = (
    data[cpm_items]
    .replace(cpm_mapping)
)

data[cpm_items] = data[cpm_items].astype("Int64")

# PtP questionnaire items (change text to numbers)
ptp_original = [f"PtP {i}" for i in range(1, 65)]

ptp_mapping = {
    "Never or hardly ever": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "About half the time": 4,
    "Often": 5,
    "Very often": 6,
    "Always or nearly always": 7
}
data[ptp_original] = (
    data[ptp_original]
    .replace(ptp_mapping)
)

data[ptp_original] = data[ptp_original].astype("Int64")

# Convert text data to numeric
def convert_to_numeric(data, columns):
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    
    data[columns] = data[columns].astype("Int64")
    
    return data

numeric_columns = ["Age", "Time", "Months", "Years"]

data = convert_to_numeric(data, numeric_columns)

# Exclusion criteria
def exclude(data, condition, description):
    n_excluded = condition.sum()
    data = data[~condition].copy()
    
    print(f"{description}: {n_excluded}")
    print(f"Remaining sample: {len(data)}\n")
    
    return data

# Initial sample
print("Initial sample:", len(data))

# Exclusion criteria
data = exclude(
    data,
    data["Age"].isna(),
    "Missing age"
)

data = exclude(
    data,
    data["Age"] < 18,
    "Under 18"
)

data = exclude(
    data,
    data["Founder"].isna(),
    "Missing founder response"
)

data = exclude(
    data,
    data["Founder"] == "No",
    "Non-founders"
)
data["CPM_completed"] = data[cpm_items].notna().sum(axis=1)
data["PtP_completed"] = data[ptp_original].notna().sum(axis=1)

# Didn't start either inventory
didnt_start_either = (
    (data["CPM_completed"] == 0) &
    (data["PtP_completed"] == 0)
)

data = exclude(
    data,
    didnt_start_either,
    "Didn't start either inventory"
)

# Completed only one inventory
only_one_inventory = (
    ((data["CPM_completed"] > 0) & (data["PtP_completed"] == 0)) |
    ((data["CPM_completed"] == 0) & (data["PtP_completed"] > 0))
)

data = exclude(
    data,
    only_one_inventory,
    "Completed only one inventory"
)

print("Final sample:", len(data))

# Reverse scoring PtP items
# Items that need to be reverse scored
reverse_ptp_items = [
    "PtP 5", "PtP 6", "PtP 7", "PtP 8",
    "PtP 11",
    "PtP 13", "PtP 14", "PtP 15", "PtP 16", "PtP 17",
    "PtP 18", "PtP 19",
    "PtP 24", "PtP 25",
    "PtP 30", "PtP 31", "PtP 32",
    "PtP 34",
    "PtP 37",
    "PtP 39", "PtP 40", "PtP 41", "PtP 42", "PtP 43", "PtP 44",
    "PtP 48", "PtP 49", "PtP 50", "PtP 51", "PtP 52",
    "PtP 56", "PtP 57", "PtP 58",
    "PtP 61"
]

# Function for reverse scoring
def reverse_score(data, reverse_items, max_score):
    scored = data.copy()
    scored[reverse_items] = max_score + 1 - data[reverse_items]
    return scored

# Reverse score the selected items
ptp_original = data[[f"PtP {i}" for i in range(1, 65)]].copy()
ptp_scored = reverse_score(
    ptp_original,
    reverse_ptp_items,
    7
)

# Check the result
print("Original PtP data:")
print(ptp_original.head())

print("\nReverse-scored PtP data:")
print(ptp_scored.head())

# Cronbach's alpha reliability for PtP
ptp_items = ptp_scored.copy()
while True:
    # Current alpha
    alpha = pg.cronbach_alpha(ptp_items)[0].item()

    print(f"\nItems: {len(ptp_items.columns)} | Alpha: {alpha:.3f}")

    # Stop when alpha reaches .70
    if alpha >= 0.70:
        print("Alpha has reached .70. Stopping.")
        break

    # Find items that meet removal criteria
    candidates = {}
    for item in ptp_items.columns:

        # Remove the item temporarily
        remaining = ptp_items.drop(columns=item)

        # Corrected item-total correlation
        total = remaining.sum(axis=1)
        r = sp.stats.pearsonr(
            ptp_items[item],
            total
        )[0].item()

        # Alpha if item deleted
        alpha_deleted = pg.cronbach_alpha(
            remaining
        )[0].item()

        # Item is considered for removal if:
        # 1. Its corrected item-total correlation is below .30
        # 2. Removing it increases Cronbach's alpha
        if r < 0.30 and alpha_deleted > alpha:
            candidates[item] = (r, alpha_deleted)

    # Stop if no items qualify for removal
    if not candidates:
        print("No items meet the removal criteria. Stopping.")
        break

    # Remove the item that produces the largest increase in alpha
    item = max(
        candidates,
        key=lambda x: candidates[x][1]
    )

    r, alpha_deleted = candidates[item]

    print(
        f"Removing {item}: "
        f"r = {r:.3f}, "
        f"alpha if deleted = {alpha_deleted:.3f}"
    )
    # Permanently remove the selected item
    ptp_items = ptp_items.drop(columns=item)


# Final results
final_alpha = pg.cronbach_alpha(ptp_items)[0].item()

print("\nFinal results")
print("Items retained:", len(ptp_items.columns))
print("Final alpha:", round(final_alpha, 3))
print("Retained items:", list(ptp_items.columns))

ptp_final = ptp_items.copy()

# Moderator variable
# Define the R&D classifications
industry_classification = {
    "Agriculture": "lRD",
    "Forestry": "lRD",
    "Mining": "lRD",
    "Energy and water supply": "lRD",
    "Waste/Recycling": "lRD",
    "Construction": "lRD",
    "Trades e.g. Plumbing/Electrical": "lRD",
    "Manufacturing": "hRD",
    "Wholesale trade": "lRD",
    "Retail trade": "lRD",
    "Real estate": "lRD",
    "Hospitality": "lRD",
    "Delivery services": "lRD",
    "Transport/Logistics": "lRD",
    "Cleaning/Maintenance": "lRD",
    "Technical services": "mRD",
    "Travel/Tourism": "lRD",
    "Entertainment": "lRD",
    "Recreation": "lRD",
    "Arts": "lRD",
    "Education/Training": "mRD",
    "Research/Academia": "hRD",
    "Health care": "mRD",
    "Social services": "mRD",
    "Administration": "lRD",
    "IT/Telecommunication": "mRD",
    "Finance/Insurance": "lRD",
    "Environment/Sustainability": "lRD"
}
# Split multiple industries for each participant
data["Industry_list"] = data["Industry.1"].str.split(",")

# Classify each selected industry
data["RD_categories"] = data["Industry_list"].apply(
    lambda industries: [
        industry_classification[industry.strip()]
        for industry in industries
        if industry.strip() in industry_classification
    ] if isinstance(industries, list) else []
)

# Count number of lRD, mRD, hRD industries selected
data["lRD_count"] = data["RD_categories"].apply(lambda x: x.count("lRD"))
data["mRD_count"] = data["RD_categories"].apply(lambda x: x.count("mRD"))
data["hRD_count"] = data["RD_categories"].apply(lambda x: x.count("hRD"))

# Calculate weighted R&D intensity score
data["RDweight"] = (
    (data["lRD_count"] * 1) +
    (data["mRD_count"] * 2) +
    (data["hRD_count"] * 3)
) / (
    data["lRD_count"] +
    data["mRD_count"] +
    data["hRD_count"]
)

# Create final moderator variable
# RDweight < 2 = lRD
# RDweight >= 2 = hRD

data["RD_moderator"] = data["RDweight"].apply(
    lambda x: pd.NA if pd.isna(x)
    else "lRD" if x < 2.00
    else "hRD"
)

# Check final groups
print(data["RD_moderator"].value_counts(dropna=False))

# Variables to summarise (excluding Industry.1)
demographic_vars = [
    "Gender",
    "Education",
    "Origin",
    "Residence",
    "Founder-Status",
    "Months",
    "Years"
]

# Frequency function
def frequency(column_name):
    freq = data[column_name].value_counts(dropna=False)
    percent = data[column_name].value_counts(normalize=True, dropna=False) * 100

    return pd.DataFrame({
        "Frequency (n)": freq,
        "Percentage (%)": percent.round(2)
    })


# Multiple-response frequency function
def multiple_response_frequency(column_name):
    freq = (
        data[column_name]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
    )

    percent = freq / len(data) * 100

    return pd.DataFrame({
        "Frequency (n)": freq,
        "Percentage (%)": percent.round(2)
    })

# Demographic frequency tables
for var in demographic_vars:
    print("\n====================")
    print(var)
    print("====================")
    print(frequency(var))

# Industry frequency table
print("\n====================")
print("Industry")
print("====================")
print(multiple_response_frequency("Industry.1"))

# CPM item frequency tables
cpm_frequency_summary = pd.DataFrame()

for item in cpm_items:
    freq = data[item].value_counts(dropna=False).sort_index()
    freq["Total responses"] = data[item].notna().sum()
    cpm_frequency_summary[item] = freq

print("\n====================")
print("CPM item frequencies")
print("====================")
print(cpm_frequency_summary)

# PtP item frequency tables
ptp_frequency_summary = pd.DataFrame()

for item in ptp_final.columns:
    freq = data[item].value_counts(dropna=False).sort_index()
    freq["Total responses"] = data[item].notna().sum()
    ptp_frequency_summary[item] = freq

print("\n====================")
print("PtP item frequencies")
print("====================")
print(ptp_frequency_summary)


# R&D moderator frequency table
print("\n====================")
print("R&D moderator")
print("====================")
print(frequency("RD_moderator"))

# Missingness check
# Count item-level missing responses

cpm_missing = data[cpm_items].isna().sum().sum()
ptp_missing = ptp_final.isna().sum().sum()
rd_missing = data["RD_moderator"].isna().sum()

# Total possible responses
cpm_total_possible = data[cpm_items].size
ptp_total_possible = ptp_final.size
rd_total_possible = len(data)

# Create summary table
missing_summary = pd.DataFrame({
    "Variable": [
        "CPM",
        "PtP",
        "RD moderator",
        "Total"
    ],
    "Missing responses (n)": [
        cpm_missing,
        ptp_missing,
        rd_missing,
        cpm_missing + ptp_missing + rd_missing
    ],
    "Possible responses (n)": [
        cpm_total_possible,
        ptp_total_possible,
        rd_total_possible,
        cpm_total_possible + ptp_total_possible + rd_total_possible
    ],
    "Missing (%)": [
        round(cpm_missing / cpm_total_possible * 100, 2),
        round(ptp_missing / ptp_total_possible * 100, 2),
        round(rd_missing / rd_total_possible * 100, 2),
        round(
            (cpm_missing + ptp_missing + rd_missing) /
            (cpm_total_possible + ptp_total_possible + rd_total_possible) * 100,
            2
        )
    ]
})

print(missing_summary)


# Missingness / Little's MCAR test

ptp_columns = [f"PtP {i}" for i in range(1, 65)]

mcar_data = data[cpm_items + ptp_columns]

mcar_test = MCARTest(method="little")
mcar_p = mcar_test(mcar_data)

print("Little's MCAR test p-value:", mcar_p)


# Show variables with missing values

missing_by_item = mcar_data.isna().sum().sort_values(ascending=False)

print(missing_by_item[missing_by_item > 0])


# Show participants with missing responses

missing_by_person = mcar_data.isna().sum(axis=1)

print(missing_by_person[missing_by_person > 0])


# View the missing items for participant 30

print(
    data.loc[30, mcar_data.columns]
    [data.loc[30, mcar_data.columns].isna()]
)

# Descriptive statistics
def descriptives(data, variables, group=None):

    if group is None:
        return data[variables].agg(
            ["count", "mean", "std", "min", "max"]
        ).T.round(2)

    return data.groupby(group)[variables].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(2)


# Calculate scale scores
data["CPM_total"] = data[cpm_items].mean(axis=1)
data["PtP_total"] = ptp_final.mean(axis=1)


# Variables to describe
inventory_vars = ["CPM_total", "PtP_total"]


# Overall descriptives
print("Overall:")
print(descriptives(data, inventory_vars))


# Descriptives by R&D group
print("\nBy R&D group:")
print(descriptives(data, inventory_vars, "RD_moderator"))

#Assumption testing
def check_normality(data, variables, group_variable=None):

    if group_variable is None:
        groups = [("All participants", data)]
    else:
        groups = data.groupby(group_variable, dropna=False)

    for group_name, group_data in groups:

        print("\n================================")
        print(group_name)
        print("================================")

        for var in variables:

            values = group_data[var].dropna()

            skewness = skew(values)

            standard_error = np.sqrt(6 / len(values))
            z_skew = skewness / standard_error

            shapiro_stat, shapiro_p = shapiro(values)

            ks = stats.kstest(
                values,
                "norm",
                args=(values.mean(), values.std())
            )

            print(f"\n{var}")
            print(f"n = {len(values)}")
            print(f"Skewness: {skewness:.3f}")
            print(f"Z-skewness: {z_skew:.3f}")
            print(
                f"Shapiro-Wilk W: "
                f"{shapiro_stat:.3f}, p: {shapiro_p:.3f}"
            )
            print(
                f"Kolmogorov-Smirnov D: "
                f"{ks.statistic:.3f}, p: {ks.pvalue:.3f}"
            )

            # Box plot
            plt.boxplot(values)
            plt.ylabel(var)
            plt.title(f"Box Plot of {var} - {group_name}")
            plt.show()

            # Q-Q plot
            stats.probplot(values, dist="norm", plot=plt)
            plt.title(f"Q-Q Plot of {var} - {group_name}")
            plt.show()

normality_vars = [
    "CPM_total",
    "PtP_total"
]

check_normality(
    data,
    normality_vars,
    "RD_moderator"
)

# Linear regression analysis
def regression_with_diagnostics(
    df,
    y="PtP_total",
    x="CPM_total",
    moderator="RD_moderator"
):

    # Prepare data
    df2 = df[[y, x, moderator]].dropna().copy()

    # Make outcome and predictor numeric
    df2[x] = pd.to_numeric(
        df2[x],
        errors="coerce"
    )

    df2[y] = pd.to_numeric(
        df2[y],
        errors="coerce"
    )

    # Code moderator
    mod_dummies = pd.get_dummies(
        df2[moderator],
        drop_first=True,
        dtype=int
    )

    mod_name = mod_dummies.columns[0]

    df2[mod_name] = (
        mod_dummies.iloc[:, 0]
        .astype(float)
    )

    # Mean-centre CPM
    df2["CPM_c"] = (
        df2[x] - df2[x].mean()
    )

    # Interaction
    df2["interaction"] = (
        df2["CPM_c"] *
        df2[mod_name]
    )

    # Predictors
    X = df2[
        ["CPM_c", mod_name, "interaction"]
    ]

    # Force everything to numeric
    X_const = sm.add_constant(X).astype(float)
    y_values = df2[y].astype(float)

    # Fit regression
    model = sm.OLS(
        y_values,
        X_const
    ).fit()

    # Influence statistics
    influence = model.get_influence()

    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    std_resid = influence.resid_studentized_internal
    studentized_deleted = (
        influence.resid_studentized_external
    )

    # Mahalanobis distance
    X_numeric = X.astype(float)

    X_centered = X_numeric - X_numeric.mean()

    covariance = np.cov(
        X_centered.to_numpy(dtype=float),
        rowvar=False
    )

    inverse_covariance = np.linalg.pinv(
        covariance
    )

    mahalanobis = np.einsum(
        "ij,jk,ik->i",
        X_centered.to_numpy(dtype=float),
        inverse_covariance,
        X_centered.to_numpy(dtype=float)
    )

    # VIF and tolerance
    vif_table = []

    for i, column in enumerate(X.columns):

        vif = variance_inflation_factor(
            X.values.astype(float),
            i
        )

        vif_table.append({
            "Variable": column,
            "VIF": vif,
            "Tolerance": 1 / vif
        })

    vif_table = pd.DataFrame(
        vif_table
    ).set_index("Variable")

    # Durbin-Watson
    dw = durbin_watson(model.resid)

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(
        df2[x],
        df2[y]
    )

    # Casewise diagnostics
    diagnostics = pd.DataFrame({

        "Predicted": model.fittedvalues,

        "Residual": model.resid,

        "Standardised residual":
            std_resid,

        "Studentized deleted residual":
            studentized_deleted,

        "Leverage":
            leverage,

        "Cook's distance":
            cooks_d,

        "Mahalanobis distance":
            mahalanobis

    }, index=df2.index)

    # ---------------------------------
    # REGRESSION ASSUMPTION PLOTS
    # ---------------------------------
    # 1. Q-Q plot — normality of residuals
    sm.qqplot(
        model.resid,
        line="45",
        fit=True
    )
    plt.title(
        "Q-Q Plot of Regression Residuals"
    )
    plt.xlabel(
        "Theoretical Quantiles"
    )
    plt.ylabel(
        "Standardised Residuals"
    )
    plt.show()

    # 2. Residuals vs fitted values
    # Checks linearity and homoscedasticity
    plt.scatter(
        model.fittedvalues,
        model.resid
    )
    plt.axhline(
        y=0,
        linestyle="--"
    )
    plt.title(
        "Residuals vs Fitted Values"
    )
    plt.xlabel(
        "Fitted Values"
    )
    plt.ylabel(
        "Residuals"
    )
    plt.show()

    return {
        "model": model,
        "diagnostics": diagnostics,
        "vif": vif_table,
        "durbin_watson": dw,
        "pearson_r": (
            pearson_r,
            pearson_p
        )
    }

# Run regression
results = regression_with_diagnostics(data)

# Regression results
print(
    results["model"].summary()
)

# Casewise diagnostics
print("\nCasewise diagnostics:")

print(
    results["diagnostics"]
)

# VIF and tolerance
print("\nVIF and Tolerance:")

print(
    results["vif"]
)

# Durbin-Watson
print(
    "\nDurbin-Watson:",
    round(
        results["durbin_watson"],
        3
    )
)

# Pearson's correlation
r, p = results["pearson_r"]

print(
    f"\nPearson's r: {r:.3f}, p = {p:.3f}"
)

results = regression_with_diagnostics(data)

print("\nPearson's r by RD group:")

for group in data["RD_moderator"].dropna().unique():

    group_data = data[
        data["RD_moderator"] == group
    ][["CPM_total", "PtP_total"]].dropna()

    r, p = pearsonr(
        group_data["CPM_total"],
        group_data["PtP_total"]
    )

    print(
        f"{group}: r = {r:.3f}, p = {p:.3f}, n = {len(group_data)}"
    )

# PROCESS-style moderation analysis
def process_moderation(
    df,
    y="PtP_total",
    x="CPM_total",
    moderator="RD_moderator"
):

    # Prepare data
    df2 = df[[y, x, moderator]].dropna().copy()

    # Make outcome and predictor numeric
    df2[x] = pd.to_numeric(df2[x], errors="coerce")
    df2[y] = pd.to_numeric(df2[y], errors="coerce")

    # Remove any rows made missing by conversion
    df2 = df2.dropna()

    # Create moderator dummy
    mod_dummies = pd.get_dummies(
        df2[moderator],
        drop_first=True,
        dtype=float
    )

    mod_name = mod_dummies.columns[0]

    df2[mod_name] = mod_dummies.iloc[:, 0].astype(float)

    # Mean-centre CPM
    df2["CPM_c"] = (
        df2[x] - df2[x].mean()
    ).astype(float)

    # Interaction
    df2["CPM_x_RD"] = (
        df2["CPM_c"] * df2[mod_name]
    ).astype(float)

    # Predictors
    X = df2[
        ["CPM_c", mod_name, "CPM_x_RD"]
    ].astype(float)

    X = sm.add_constant(X).astype(float)

    Y = df2[y].astype(float)

    # Main moderation model
    model = sm.OLS(Y, X).fit()

    # Regression coefficients
    confidence_intervals = model.conf_int()

    coefficients = pd.DataFrame({
        "B": model.params,
        "SE": model.bse,
        "t": model.tvalues,
        "p": model.pvalues,
        "LLCI": confidence_intervals[0],
        "ULCI": confidence_intervals[1]
    })

    # Conditional effects of CPM within each RD group
    simple_slopes = []

    for group in df2[moderator].unique():

        group_data = df2[
            df2[moderator] == group
        ].copy()

        # Force variables to numeric
        group_y = group_data[y].astype(float)
        group_x = group_data[["CPM_c"]].astype(float)

        group_X = sm.add_constant(group_x).astype(float)

        group_model = sm.OLS(
            group_y,
            group_X
        ).fit()

        ci = group_model.conf_int()

        simple_slopes.append({
            "RD group": group,
            "Effect of CPM": group_model.params["CPM_c"],
            "SE": group_model.bse["CPM_c"],
            "t": group_model.tvalues["CPM_c"],
            "p": group_model.pvalues["CPM_c"],
            "LLCI": ci.loc["CPM_c", 0],
            "ULCI": ci.loc["CPM_c", 1],
            "n": len(group_data)
        })

    simple_slopes = pd.DataFrame(simple_slopes)

    return {
        "model": model,
        "coefficients": coefficients,
        "simple_slopes": simple_slopes
    }

results = process_moderation(data)

print("\n==============================")
print("MODERATION ANALYSIS")
print("==============================")

print("\nRegression coefficients:")
print(results["coefficients"].round(3))

print("\nConditional effects of CPM by RD group:")
print(results["simple_slopes"].round(3))

print("\nFull regression model:")
print(results["model"].summary())