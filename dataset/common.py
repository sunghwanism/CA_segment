
import ast
import pandas as pd
import pydicom

from utils.functions import load_config
cfg = load_config()

def load_patient_data(PATH):
    df = pd.read_csv(PATH)
    
    if "FolderList" in df.columns:
        df['FolderList'] = df['FolderList'].apply(ast.literal_eval)
    
    if "PatientID" in df.columns:
        df['PatientID'] = df['PatientID'].astype(int)

    if 'Direction' in df.columns:
        df['Direction'] = df['Direction'].apply(ast.literal_eval)

    return df

def load_scan(FileList):
    slices = [pydicom.dcmread(cfg.paths.raw_data + s) for s in FileList]
    
    return slices


