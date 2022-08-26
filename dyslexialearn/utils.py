import nibabel as nib
import pandas as pd
import json
import numpy as np

def load_data():
    csv = pd.read_csv("data/n192_data_for_resid.csv") 
    mask = nib.load("data/brain_mask.nii").get_fdata()

    X = []
    for id_name in csv['id']:
        filename = f"data/raw_images/jac_rc1r{id_name}_debiased_deskulled_denoised_xTemplate_subtypes.nii"
        X.append(nib.load(filename).get_fdata())

    return csv, np.array(X), mask
