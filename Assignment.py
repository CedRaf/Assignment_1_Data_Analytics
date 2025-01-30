import numpy as np
import pandas as pd
from datetime import datetime, timedelta

values = [0,1,2,3,4,5,6,7,8,9]

np.random.seed(50)
n_samples = 100

treatment_status = np.random.choice([0,1], n_samples, [0.6,0.4])
treatment_time = np.where(treatment_status==1, np.random.randint(5, 20, n_samples), np.nan)

random_year = np.random.randint(2020,2025,n_samples)
random_month = np.random.randint(1,13,n_samples)

#This shit is quite questionable bc it was implemented differently in the actual study
ds = pd.DataFrame({
    'id': range(n_samples),
    'age': np.round(np.random.normal(50, 10, n_samples)),
    'admitted_date':  pd.to_datetime({"year": random_year, "month": random_month, "day": 1}).dt.strftime("%Y-%m"),
    'sex': np.random.choice([0,1], n_samples), # male = 1, female = 0
    'treatment_status': treatment_status,      # yes = 1, no = 0
    'treatment_time': treatment_time,
    'pain_baseline': np.random.choice(values, n_samples),
    'urgency_baseline': np.random.choice(values, n_samples),
    'frequency_baseline': np.random.choice(values, n_samples),
    'pain_treatment': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'urgency_treatment': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
    'frequency_treatment': np.where(treatment_status==1, np.random.choice(values, n_samples), np.nan),
})

print(ds.head())

#risk sets
#we only select those who have not received treatment bc they are our "control group" at "risk" of receiving treatment
def risk_set(ds, treatment_time):
    return ds[(ds["treatment_status"]== 0) | (ds["treatment_time"] > treatment_time)]

sample_risk_set = risk_set(ds, 10)
#print(sample_risk_set.head())

#creating quantiles
def create_tercile(column):
    return pd.qcut(column.dropna(), 3, [0,1,2]) 

ds["painB_tercile"] = create_tercile(ds["pain_baseline"])
ds["urgencyB_tercile"] = create_tercile(ds["urgency_baseline"])
ds["frequencyB_tercile"] = create_tercile(ds["frequency_baseline"])
ds["painT_tercile"] = create_tercile(ds["pain_treatment"])
ds["urgencyT_tercile"] = create_tercile(ds["urgency_treatment"])
ds["frequencyT_tercile"] = create_tercile(ds["frequency_treatment"])

#print(ds[["pain_baseline", "painB_tercile", "urgency_baseline", "urgencyB_tercile", "frequency_baseline", "frequencyB_tercile"]].head())
#print(ds[["pain_treatment", "painT_tercile", "urgency_treatment", "urgencyT_tercile", "frequency_treatment", "frequencyT_tercile"]].head())