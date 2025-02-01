import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

values = np.arange(0,10)

np.random.seed(50)
n_samples = 100

treatment_status = np.random.choice([0,1], n_samples, [0.4,0.6])

admitted_date = pd.to_datetime({
    "year": np.random.randint(2020,2023, n_samples),
    "month": np.random.randint(1,13,n_samples),
    "day": 1
})

def add_random_months(date):
    return date + relativedelta(months=np.random.randint(1,24))

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
    'sex': np.random.choice([0,1], n_samples), 
    'treatment_status': treatment_status,      
    'treatment_time': treatment_time,
    'pain_baseline': np.round(np.random.normal(5,3,n_samples)),
    'urgency_baseline': np.round(np.random.normal(5,2,n_samples)),
    'frequency_baseline': np.round(np.random.normal(3,2,n_samples)),
    'painT_during': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'urgencyT_during': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'frequencyT_during': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'painT_3months': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'urgencyT_3months': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'frequency_3months': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'painT_6months': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'urgencyT_6months': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'frequency_6months': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
})

#print(ds.head())

#risk sets
#select those who have not yet received at the time a patient received treatment (generates a list of all potential pairings)
def risk_set(ds):
    risk_sets = {}
    treated_patients = ds[ds["treatment_status"]==1]

    for _, treated in treated_patients.iterrows():
        treated_id = treated["id"]
        treatment_time = treated["treatment_time"]

        controls = ds[(ds["treatment_time"] > treatment_time) | (ds["treatment_status"]==0)]
        controls = controls[controls["admitted_date"] <= treatment_time]

        risk_sets[treated_id] = controls

    return risk_sets


sample_risk_set = risk_set(ds)
treated_id = list(sample_risk_set.keys())[1]  #change index value to get the list of possible controls for the treated patient
#print(f"Risk Set for Treated Patient {treated_id}:")
#print(sample_risk_set[treated_id])

#creating quantiles
def create_tercile(column):
    bins = [0, 3, 6, 9]  # Define bin edges
    labels = [0, 1, 2]   # Assign tercile labels
    return pd.cut(column, bins=bins, labels=labels, include_lowest=True)

ds["painB_tercile"] = create_tercile(ds["pain_baseline"])
ds["urgencyB_tercile"] = create_tercile(ds["urgency_baseline"])
ds["frequencyB_tercile"] = create_tercile(ds["frequency_baseline"])
ds["painT_tercile"] = create_tercile(ds["painT_during"])
ds["urgencyT_tercile"] = create_tercile(ds["urgencyT_during"])
ds["frequencyT_tercile"] = create_tercile(ds["frequencyT_during"])

#print(ds[["pain_baseline", "painB_tercile", "urgency_baseline", "urgencyB_tercile", "frequency_baseline", "frequencyB_tercile"]].head())
#print(ds[["painT_during", "painT_tercile", "urgencyT_during", "urgencyT_tercile", "frequencyT_during", "frequencyT_tercile"]].head())