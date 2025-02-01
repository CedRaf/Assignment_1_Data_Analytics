import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore

values = np.arange(0, 10)

np.random.seed(50)
n_samples = 100

treatment_status = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])

admitted_date = pd.to_datetime({
    "year": np.random.randint(2020, 2023, n_samples),
    "month": np.random.randint(1, 13, n_samples),
    "day": 1
})

def add_random_months(date):
    return date + relativedelta(months=np.random.randint(1, 24))

treatment_time = pd.Series(np.nan, index=range(n_samples))  # Initialize with NaT (Not a Time)
treatment_time = np.where(
    treatment_status == 1,
    [add_random_months(date) for date in admitted_date],
    pd.NaT
)
admitted_date = admitted_date.dt.strftime("%Y-%m")
treatment_time = pd.Series(treatment_time).dt.strftime("%Y-%m")


# male=1, female=0 | treated=1, not treated=0 | painT_during = pain during treatment | painT_3months = pain 3 months post treatment
ds = pd.DataFrame({
    'id': range(n_samples),
    'age': np.round(np.random.normal(50, 10, n_samples)),
    'admitted_date': admitted_date,
    'sex': np.random.choice([0, 1], n_samples),
    'treatment_status': treatment_status,
    'treatment_time': treatment_time,

    # Baseline values (random normal distribution)
    'pain_baseline': np.round(np.random.normal(5, 3, n_samples)),
    'urgency_baseline': np.round(np.random.normal(5, 2, n_samples)),
    'frequency_baseline': np.round(np.random.normal(3, 2, n_samples)),
})

# Ensure all baseline values are non-negative
ds[['pain_baseline', 'urgency_baseline', 'frequency_baseline']] = ds[['pain_baseline', 'urgency_baseline', 'frequency_baseline']].clip(lower=0)

# Apply the decreasing pattern where values drop over time
ds['painT_during'] = np.where(ds['treatment_status'] == 1, ds['pain_baseline'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)
ds['urgencyT_during'] = np.where(ds['treatment_status'] == 1, ds['urgency_baseline'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)
ds['frequencyT_during'] = np.where(ds['treatment_status'] == 1, ds['frequency_baseline'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)

ds['painT_3months'] = np.where(ds['treatment_status'] == 1, ds['painT_during'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)
ds['urgencyT_3months'] = np.where(ds['treatment_status'] == 1, ds['urgencyT_during'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)
ds['frequency_3months'] = np.where(ds['treatment_status'] == 1, ds['frequencyT_during'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)

ds['painT_6months'] = np.where(ds['treatment_status'] == 1, ds['painT_3months'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)
ds['urgencyT_6months'] = np.where(ds['treatment_status'] == 1, ds['urgencyT_3months'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)
ds['frequency_6months'] = np.where(ds['treatment_status'] == 1, ds['frequency_3months'] * np.random.uniform(0.7, 0.9, n_samples), np.nan)

# Ensure values do not go below 0
ds[['painT_during', 'urgencyT_during', 'frequencyT_during',
    'painT_3months', 'urgencyT_3months', 'frequency_3months',
    'painT_6months', 'urgencyT_6months', 'frequency_6months']] = \
    ds[['painT_during', 'urgencyT_during', 'frequencyT_during',
        'painT_3months', 'urgencyT_3months', 'frequency_3months',
        'painT_6months', 'urgencyT_6months', 'frequency_6months']].clip(lower=0).round()


selected_columns = [
    'id',
    'pain_baseline', 'urgency_baseline', 'frequency_baseline',
    'painT_during', 'urgencyT_during', 'frequencyT_during',
    'painT_3months', 'urgencyT_3months', 'frequency_3months',
    'painT_6months', 'urgencyT_6months', 'frequency_6months'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#print(ds[selected_columns].head(10))

# Create binary variables for terciles
def create_tercile(column):
    return pd.qcut(column.dropna(), 3, [0, 1, 2])

# Apply terciles for baseline and treatment symptom variables
ds["painB_tercile"] = create_tercile(ds["pain_baseline"])
ds["urgencyB_tercile"] = create_tercile(ds["urgency_baseline"])
ds["frequencyB_tercile"] = create_tercile(ds["frequency_baseline"])
ds["painT_tercile"] = create_tercile(ds["painT_during"])
ds["urgencyT_tercile"] = create_tercile(ds["urgencyT_during"])
ds["frequencyT_tercile"] = create_tercile(ds["frequencyT_during"])

# Step 2: Create binary variables for each tercile
def create_tercile_binaries(df, column_name):
    """Create binary variables for each tercile (low, middle, high)."""
    df[f"{column_name}_tercile_1"] = (df[column_name] == 0).astype(int)
    df[f"{column_name}_tercile_2"] = (df[column_name] == 1).astype(int)
    df[f"{column_name}_tercile_3"] = (df[column_name] == 2).astype(int)
    return df

# Apply binary columns for each tercile group
ds = create_tercile_binaries(ds, "pain_baseline")
ds = create_tercile_binaries(ds, "urgency_baseline")
ds = create_tercile_binaries(ds, "frequency_baseline")
ds = create_tercile_binaries(ds, "painT_during")
ds = create_tercile_binaries(ds, "urgencyT_during")
ds = create_tercile_binaries(ds, "frequencyT_during")

# Step 3: Define the risk set based on tercile balance
def balanced_risk_set(treated_patient, ds):
    """Generate a risk set for the treated patient, considering tercile balancing and treatment date."""
    # Get the tercile of the treated patient for each symptom variable
    treated_terciles = {
        "painB_tercile": treated_patient["painB_tercile"],
        "urgencyB_tercile": treated_patient["urgencyB_tercile"],
        "frequencyB_tercile": treated_patient["frequencyB_tercile"],
        "painT_tercile": treated_patient["painT_tercile"],
        "urgencyT_tercile": treated_patient["urgencyT_tercile"],
        "frequencyT_tercile": treated_patient["frequencyT_tercile"]
    }

    # Get the treatment time of the treated patient
    treatment_time = treated_patient["treatment_time"]

    # Filter out controls (those with treatment time later than the treated patient)
    controls = ds[(ds["treatment_time"] > treatment_time) | (ds["treatment_status"]==0)]
    
    # Filter the controls based on tercile matching
    for key, value in treated_terciles.items():
        controls = controls[controls[key] == value]

    # Apply date check: Only include controls where admitted_date <= treatment_time
    controls = controls[controls["admitted_date"] <= treatment_time]

    return controls

def risk_set(treated_patient, ds):
    risk_sets = {}
    treated_patients = ds[ds["treatment_status"]==1]

    for _, treated in treated_patients.iterrows():
        treated_id = treated["id"]
        treatment_time = treated["treatment_time"]

        controls = ds[(ds["treatment_time"] > treatment_time) | (ds["treatment_status"]==0)]
        controls = controls[controls["admitted_date"] <= treatment_time]

        risk_sets[treated_id] = controls

    return risk_sets

#Difference from original, og code performed risk set on all treated patients at once, mine does it on the parametered patient and is called multiple times

# Step 4: Calculate the Mahalanobis distance
def mahalanobis_distance(treated_row, control_row, inv_cov_matrix):
    """Calculate Mahalanobis distance between treated and control patient."""
    treated_vector = treated_row[["pain_baseline", "urgency_baseline", "frequency_baseline", "painT_during", "urgencyT_during", "frequencyT_during"]]
    control_vector = control_row[["pain_baseline", "urgency_baseline", "frequency_baseline", "painT_during", "urgencyT_during", "frequencyT_during"]]
    
    return mahalanobis(treated_vector, control_vector, inv_cov_matrix)

# Step 5: Matching function
def find_best_match(treated_row, risk_set, inv_cov_matrix):
    """Find the best match for the treated patient from the risk set."""
    if not risk_set.empty:
        distances = risk_set.apply(lambda row: mahalanobis_distance(treated_row, row, inv_cov_matrix), axis=1)
        best_match = risk_set.iloc[distances.idxmin()]
        return best_match
    else:
        return None

# Step 6: Covariance matrix and inverse covariance matrix
# Standardize the variables before calculating the covariance matrix
#ds[["pain_baseline", "urgency_baseline", "frequency_baseline", "painT_during", "urgencyT_during", "frequencyT_during"]] = ds[["pain_baseline", "urgency_baseline", "frequency_baseline", "painT_during", "urgencyT_during", "frequencyT_during"]].dropna().apply(zscore)

cov_matrix = np.cov(ds[["pain_baseline", "urgency_baseline", "frequency_baseline", "painT_during", "urgencyT_during", "frequencyT_during"]].dropna(), rowvar=False)
if np.linalg.det(cov_matrix) == 0:
    print("Covariance matrix is singular, cannot calculate inverse.")
else:
    inv_cov_matrix = np.linalg.inv(cov_matrix)

# Example: Find a match for a treated patient
treated_patient = ds[ds["treatment_status"] == 1].iloc[5]  # Choose the first treated patient
risk_set_for_treated = risk_set(treated_patient, ds)
#best_control_match = find_best_match(treated_patient, risk_set_for_treated, inv_cov_matrix)

first_key = list(risk_set_for_treated.keys())[0]  # Get the first key
df = risk_set_for_treated[first_key]  # Extract the DataFrame
print(df[["pain_baseline", "painB_tercile", "urgency_baseline", "urgencyB_tercile", "frequency_baseline", "frequencyB_tercile"]].head())



# treated_patients = ds[ds['treatment_status'] == 1][['id', 'age', 'treatment_status']]
# print(treated_patients)
# untreated_patients = ds[ds['treatment_status'] == 0][['id', 'age', 'treatment_status']]
# print(untreated_patients)


# Print the matched control patient
#if best_control_match is not None:
    #print("Matched Control Patient:")
    #print(best_control_match)
#else:
    #print(f"No suitable control found for treated patient {treated_patient['id']}")
