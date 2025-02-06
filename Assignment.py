import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
from numpy.linalg import inv



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
    'pain_baseline': np.clip(np.round(np.random.normal(5, 3, n_samples)), 1, 9),
    'urgency_baseline': np.clip(np.round(np.random.normal(5, 3, n_samples)), 1, 9),
    'frequency_baseline': np.clip(np.round(np.random.normal(5, 3, n_samples)), 1, 9)
})

# Ensure all baseline values are non-negative
ds[['pain_baseline', 'urgency_baseline', 'frequency_baseline']] = ds[['pain_baseline', 'urgency_baseline', 'frequency_baseline']].clip(lower=0)

# Apply the decreasing pattern where values drop over time
ds['painT_during'] =  ds['pain_baseline'] * np.random.uniform(0.7, 0.9, n_samples)
ds['urgencyT_during'] = ds['urgency_baseline'] * np.random.uniform(0.7, 0.9, n_samples)
ds['frequencyT_during'] =  ds['frequency_baseline'] * np.random.uniform(0.7, 0.9, n_samples)

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






def create_tercile(column):
    bins = [0, 3, 6, 9]  # Define bin edges
    labels = [0, 1, 10]   # Assign tercile labels
    return pd.cut(column, bins=bins, labels=labels, include_lowest=True)

# Apply terciles for each symptom variable first
ds["painB_tercile"] = create_tercile(ds["pain_baseline"])
ds["urgencyB_tercile"] = create_tercile(ds["urgency_baseline"])
ds["frequencyB_tercile"] = create_tercile(ds["frequency_baseline"])
ds["painT_tercile"] = create_tercile(ds["painT_during"])
ds["urgencyT_tercile"] = create_tercile(ds["urgencyT_during"])
ds["frequencyT_tercile"] = create_tercile(ds["frequencyT_during"])


def create_tercile_binaries(df, column_name):
    """Create binary variables for each tercile (low, middle, high)."""
    df[f"{column_name}_tercile_1"] = (df[column_name] == 0).astype(int)
    df[f"{column_name}_tercile_2"] = (df[column_name] == 1).astype(int)
    df[f"{column_name}_tercile_3"] = (df[column_name] == 2).astype(int)
    return df

# Apply the binary terciles for each symptom variable
ds = create_tercile_binaries(ds, "painB_tercile")
ds = create_tercile_binaries(ds, "urgencyB_tercile")
ds = create_tercile_binaries(ds, "frequencyB_tercile")
ds = create_tercile_binaries(ds, "painT_tercile")
ds = create_tercile_binaries(ds, "urgencyT_tercile")
ds = create_tercile_binaries(ds, "frequencyT_tercile")


selected_columns = [
    'id', 'pain_baseline', 'urgency_baseline', 'frequency_baseline',
    'painT_during', 'urgencyT_during', 'frequencyT_during',
    'painB_tercile', 'urgencyB_tercile', 'frequencyB_tercile',
    'painT_tercile', 'urgencyT_tercile', 'frequencyT_tercile'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(ds[selected_columns].head(10))

def risk_set(ds):
    """Create risk sets of untreated patients for each treated patient."""
    risk_sets = {}

    # Identify treated patients
    treated_patients = ds[ds["treatment_status"] == 1]

    # Iterate through each treated patient
    for _, treated in treated_patients.iterrows():
        treated_id = treated["id"]
        treatment_time = treated["treatment_time"]

        # Step 1: Define the Risk Set (all untreated patients up to time T)
        controls = ds[
            (ds["treatment_status"] == 0) | (ds["treatment_time"] > treatment_time)
        ]
        controls = controls[(controls["admitted_date"] <= treatment_time) & (controls["sex"] == treated_sex) & (abs(controls["age"] - treated["age"])<=5)]

        # Step 2: Store risk set (no filtering on terciles yet)
        risk_sets[treated_id] = controls

    return risk_sets


def match_patients(treated, risk_set, tercile_columns):
    """Find the best match for a treated patient using Mahalanobis distance and balanced terciles."""
    
    # Extract treated patient's tercile values
    treated_terciles = treated[tercile_columns].values.reshape(1, -1)

    # Compute Mahalanobis distance for each control in the risk set
    cov_matrix = np.cov(risk_set[tercile_columns].T)  # Covariance matrix
    inv_cov_matrix = inv(cov_matrix)  # Inverse covariance

    # Calculate distances for each control
    distances = risk_set.apply(
        lambda row: mahalanobis(row[tercile_columns].values, treated_terciles, inv_cov_matrix), 
        axis=1
    )

    # Select the closest match
    best_match = risk_set.iloc[distances.idxmin()]
    return best_match



# Define tercile-based binary variable columns
tercile_columns = [
    "painB_tercile", "urgencyB_tercile", "frequencyB_tercile",
    "painT_tercile", "urgencyT_tercile", "frequencyT_tercile"
]

# Get risk sets
risk_sets = risk_set(ds)

# Match each treated patient with a control
matched_pairs = {}
matched_controls = set()  # This will store the IDs of matched controls

for treated_id, risk in risk_sets.items():
    treated_patient = ds.loc[ds["id"] == treated_id].iloc[0]
    
    # Filter out already matched controls
    available_controls = risk[~risk['id'].isin(matched_controls)]
    
    if not available_controls.empty:
        matched_control = match_patients(treated_patient, available_controls, tercile_columns)
        
        if matched_control is not None:
            # Add the matched control's ID to the set
            matched_controls.add(matched_control['id'])
            
            # Store the matched pair (treated patient, matched control)
            matched_pairs[treated_id] = (treated_patient, matched_control)

# Print each matched pair in a more readable format
for treated_id, matched_control in matched_pairs.items():
    print(f"Treated ID: {treated_id}")
    print(matched_control)
    print("-" * 50)  # Separator between each matched pair


