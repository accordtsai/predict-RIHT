# Development and validation of a machine learning model of radiation-induced hypothyroidism with clinical and dose–volume features
# Mu-Hung Tsai et al.
# https://doi.org/10.1016/j.radonc.2023.109911

### This script illustrates predicting radiation-induced hypothyroidism risk in a single patient
### using thyroid DVH data acquired from the treatment planning system via Varian ESAPI

### Author: Mu-Hung Tsai, 2023
### Institute of Computer Science and Information Engineering, National Cheng Kung University, Tainan, Taiwan
### Department of Radiation Oncology, National Cheng Kung University Hospital, 
### College of Medicine, National Cheng Kung University, Tainan, Taiwan

# Requirements: sksurv == 0.21.0, installed and working pyesapi (must be able to connect to TPS database)
# One of these pre-trained models:
# - Cox-thyroid.pkl (980 KB)
# - RF-thyroid.pkl (56 MB)
# - GB-thyroid.pkl (36 KB)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sksurv
import pickle
import atexit
import pyesapi

# First define patient ID to extract information from; there must be an organ named 'Thyroid'
patient_id = '12345678'

# Define functions for computing DVH dose parameters
def getVolumeSummary(plan, structure):
    '''
    Returns "thyroid-only" features from DVH object:
    Thyroid_volume,Thyroid_min,Thyroid_max,Thyroid_mean,Thyroid_modal,Thyroid_median,Thyroid_sphere,
    Thyroid_V1,Thyroid_V2,Thyroid_V3,Thyroid_V4,Thyroid_V5,...,Thyroid_V75,Thyroid_V76,Thyroid_V77,Thyroid_V78,Thyroid_V79,Thyroid_V80,
    Thyroid_VS1,Thyroid_VS2,Thyroid_VS3,Thyroid_VS4,Thyroid_VS5,...,Thyroid_VS75,Thyroid_VS76,Thyroid_VS77,Thyroid_VS78,Thyroid_VS79,Thyroid_VS80,
    Thyroid_V30_60
    '''
    d = {}
    
    dvh = plan.GetDVHCumulativeData(structure, pyesapi.DoseValuePresentation.Absolute, pyesapi.VolumePresentation.Relative, 1.0)
    volume = round(structure.Volume, 2)
    
    d['Thyroid_volume'] = volume
    d['Thyroid_min'] = dvh.MinDose.Dose
    d['Thyroid_max'] = dvh.MaxDose.Dose
    d['Thyroid_mean'] = dvh.MeanDose.Dose
    d['Thyroid_median'] = dvh.MedianDose.Dose
    
    return d

def roundVolumeAtDose(structure, doselevel):
    # Dose level in Gy
    dv = pyesapi.DoseValue(doselevel * 1.0,'Gy') # Convert to float
    dose = plan.GetVolumeAtDose(structure, dv, pyesapi.VolumePresentation.Relative)
    return round(dose, 2)

def getVolAtDose(original_dict, structure):
    
    structure_volume = original_dict['Thyroid_volume']
    
    for i in range(1, 81):
        # Thyroid_Vx
        name = 'Thyroid_V' + str(i)
        volume = roundVolumeAtDose(structure, i)
        original_dict[name] = volume
    for i in range(1, 81):
        # Thyroid_VSx
        spared_name = 'Thyroid_VS' + str(i)
        volume = roundVolumeAtDose(structure, i)
        spared_volume = (100-volume) / 100 * structure_volume # Absolute volume spared from dose
        original_dict[spared_name] = spared_volume
    
    # Thyroid_V30_60
    original_dict['Thyroid_V30_60'] = roundVolumeAtDose(structure, 30) - roundVolumeAtDose(structure, 60)

    return original_dict

def getBaselineData(plan, structure):
    data = getVolumeSummary(plan, structure)
    data = getVolAtDose(data, structure)
    return data
    
# Load the pre-trained model; uncomment other models to try them
with open('models/GB-thyroid.pkl','rb') as file:
    model_gb = pickle.load(file)
#with open('RF-thyroid.pkl','rb') as file:
#    model_rf = pickle.load(file)
#with open('Cox-thyroid.pkl','rb') as file:
#    model_cox = pickle.load(file)

# Load data using ESAPI
app = pyesapi.CustomScriptExecutable.CreateApplication('predict_RIHT')
atexit.register(app.Dispose)
pat = app.OpenPatientById(patient_id)
plan = pat.CoursesLot(0).PlanSetupsLot(0)
thyroid = plan.StructureSet.StructuresLot('Thyroid')
dvhdata = getBaselineData(plan, thyroid)
x_predict = pd.DataFrame.from_dict([dvhdata])

# Compute survival function and plot
time_points = np.arange(0.15, 5, 0.05)
fig = plt.figure(figsize=(16, 8))

pred_surv_gb = model_gb.predict_survival_function(x_predict)
#pred_surv_rf = model_rf.predict_survival_function(x_predict)
#pred_surv_cox = model_cox.predict_survival_function(x_predict)

for i, surv_func in enumerate(pred_surv_gb):
    plt.step(time_points, surv_func(time_points), where="post",
             label="GB prediction")

# for i, surv_func in enumerate(pred_surv_rf):
#     plt.step(time_points, surv_func(time_points), where="post",
#              label="RF prediction")

# for i, surv_func in enumerate(pred_surv_cox):
#     plt.step(time_points, surv_func(time_points), where="post",
#              label="Cox prediction")

plt.ylim([0.0, 1])
plt.xlim(left=0)
plt.ylabel("Estimated Probability of Freedom from Grade ≥1 RIHT")
plt.xlabel("Time after Treatment")
plt.legend(loc="best")
plt.show()