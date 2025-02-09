import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
from numpy.linalg import pinv, LinAlgError
import plotly.express as px


values = np.arange(0, 10)

np.random.seed(50)
n_samples = 400

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

ds['painT_3months'] = ds['painT_during'] * np.random.uniform(0.7, 0.9, n_samples)
ds['urgencyT_3months'] = ds['urgencyT_during'] * np.random.uniform(0.7, 0.9, n_samples)
ds['frequency_3months'] = ds['frequencyT_during'] * np.random.uniform(0.7, 0.9, n_samples)

ds['painT_6months'] = ds['painT_3months'] * np.random.uniform(0.7, 0.9, n_samples)
ds['urgencyT_6months'] = ds['urgencyT_3months'] * np.random.uniform(0.7, 0.9, n_samples)
ds['frequency_6months'] = ds['frequency_3months'] * np.random.uniform(0.7, 0.9, n_samples)


# Ensure values do not go below 0
ds[['painT_during', 'urgencyT_during', 'frequencyT_during',
    'painT_3months', 'urgencyT_3months', 'frequency_3months',
    'painT_6months', 'urgencyT_6months', 'frequency_6months']] = \
    ds[['painT_during', 'urgencyT_during', 'frequencyT_during',
        'painT_3months', 'urgencyT_3months', 'frequency_3months',
        'painT_6months', 'urgencyT_6months', 'frequency_6months']].clip(lower=0).round()

def create_tercile(column):
    bins = [0, 3, 6, 9] 
    labels = [0, 1, 2]   
    return pd.cut(column, bins=bins, labels=labels, include_lowest=True)

# Apply terciles for each symptom variable first
ds["painB_tercile"] = create_tercile(ds["pain_baseline"])
ds["urgencyB_tercile"] = create_tercile(ds["urgency_baseline"])
ds["frequencyB_tercile"] = create_tercile(ds["frequency_baseline"])
ds["painT_tercile"] = create_tercile(ds["painT_during"])
ds["urgencyT_tercile"] = create_tercile(ds["urgencyT_during"])
ds["frequencyT_tercile"] = create_tercile(ds["frequencyT_during"])

selected_columns = [
    'id', 'pain_baseline', 'urgency_baseline', 'frequency_baseline',
    'painT_during', 'urgencyT_during', 'frequencyT_during',
    'painB_tercile', 'urgencyB_tercile', 'frequencyB_tercile',
    'painT_tercile', 'urgencyT_tercile', 'frequencyT_tercile'
]

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
        controls = controls[(controls["admitted_date"] <= treatment_time) & (abs(controls["age"] - treated["age"])<=5)]

        # Step 2: Store risk set (no filtering on terciles yet)
        risk_sets[treated_id] = controls

    return risk_sets


def match_patients(treated, risk_set, true_value_columns, regularization=1e-6):
    """Find the best match for a treated patient using Mahalanobis distance and balanced true values."""
    
    # Extract treated patient's true values
    treated_values = treated[true_value_columns].values.reshape(1, -1)

    # Drop NaN values from the risk set
    risk_set_clean = risk_set[true_value_columns].dropna()

    
    if risk_set_clean.shape[0] < 2:
        return None  

    # Ensure numeric conversion
    risk_set_clean = risk_set_clean.astype(float)

    # Compute covariance matrix
    cov_matrix = np.cov(risk_set_clean.T)

    # Check if covariance matrix is singular
    if np.linalg.matrix_rank(cov_matrix) < len(true_value_columns):
        print("Warning: Singular covariance matrix detected, applying regularization.")
        cov_matrix += np.eye(len(true_value_columns)) * regularization  # Regularization

    try:
        inv_cov_matrix = pinv(cov_matrix) 
    except LinAlgError:
        print("Error: Covariance matrix inversion failed.")
        return None  # Skip this patient

    # Calculate Mahalanobis distances for each control
    distances = risk_set_clean.apply(
        lambda row: mahalanobis(row.values, treated_values.flatten(), inv_cov_matrix), 
        axis=1
    )

    # Select the closest match
    best_match_index = distances.idxmin()
    best_match = risk_set.loc[best_match_index]  
    return best_match



def apply_soft_matching(risk_set, treated_terciles, tercile_columns, min_matches=4):


    # Count how many tercile values match for each control
    risk_set = risk_set.copy() 
    risk_set.loc[:, "tercile_matches"] = (risk_set[tercile_columns] == treated_terciles).sum(axis=1)

    # Keep only controls with at least min_matches
    filtered_risk_set = risk_set[risk_set["tercile_matches"] >= min_matches].copy()

    return filtered_risk_set




# Define tercile-based binary variable columns
tercile_columns = [
    "painB_tercile", "urgencyB_tercile", "frequencyB_tercile",
    "painT_tercile", "urgencyT_tercile", "frequencyT_tercile"
]
true_value_columns = [
    "pain_baseline", "urgency_baseline", "frequency_baseline",
    "painT_during", "urgencyT_during", "frequencyT_during"
]






risk_sets = risk_set(ds)

# Match each treated patient with a control
matched_pairs = {}
matched_controls = set()  # Store IDs of matched controls

for treated_id, risk in risk_sets.items():
    treated_patient = ds.loc[ds["id"] == treated_id].iloc[0]

    # Extract tercile values of the treated patient
    treated_terciles = treated_patient[tercile_columns]  

    # Filter out already matched controls
    available_controls = risk[~risk['id'].isin(matched_controls)]
    filtered_risk_set = apply_soft_matching(available_controls, treated_terciles, tercile_columns, min_matches=3)
    if not filtered_risk_set.empty:
        matched_control = match_patients(treated_patient, filtered_risk_set, true_value_columns)

        if matched_control is not None:
            matched_controls.add(matched_control['id'])
            matched_pairs[treated_id] = (treated_patient, matched_control)

matched_data = []

# Iterate through the matched pairs and store them in the list
for treated_id, (treated_patient, matched_control) in matched_pairs.items():
    # Create a dictionary for the treated patient
    treated_dict = treated_patient.to_dict()
    treated_dict['type'] = 'treated'  
    
    # Create a dictionary for the matched control
    control_dict = matched_control.to_dict()
    control_dict['type'] = 'control'  
    
    # Append both to the list
    matched_data.append(treated_dict)
    matched_data.append(control_dict)

# Convert the list of dictionaries to a DataFrame
matched_df = pd.DataFrame(matched_data)

# # Export the DataFrame to a CSV file
# matched_df.to_csv('softmatched_pairs.csv', index=False)

# print("Matched pairs exported to 'softmatched_pairs.csv'")



urgency_melted = matched_df.melt(id_vars=["id", "type"],  
                                value_vars=["urgency_baseline", "urgencyT_during", "urgencyT_3months", "urgencyT_6months"],  
                                var_name="Timepoint", 
                                value_name="Urgency Score")

pain_melted = matched_df.melt(id_vars=["id", "type"],  
                              value_vars=["pain_baseline", "painT_during", "painT_3months", "painT_6months"],  
                              var_name="Timepoint", 
                              value_name="Pain Score")

frequency_melted = matched_df.melt(id_vars=["id", "type"],  
                                   value_vars=["frequency_baseline", "frequencyT_during", "frequency_3months", "frequency_6months"],  
                                   var_name="Timepoint", 
                                   value_name="Frequency Score")


fig_urgency = px.box(urgency_melted, 
                      x="Timepoint", 
                      y="Urgency Score", 
                      color="type",  
                      title="Urgency Comparison Across Timepoints")

fig_pain = px.box(pain_melted, 
                  x="Timepoint", 
                  y="Pain Score", 
                  color="type",  
                  title="Pain Comparison Across Timepoints")

fig_frequency = px.box(frequency_melted, 
                       x="Timepoint", 
                       y="Frequency Score", 
                       color="type",  
                       title="Frequency Comparison Across Timepoints")

# Show the plots
fig_urgency.show()
fig_pain.show()
fig_frequency.show()





