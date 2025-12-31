
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import json

import re
import ast
import pandas as pd
import numpy as np
import pydicom
from collections import defaultdict

from tqdm.auto import tqdm

import yaml
from box import ConfigBox

from utils.functions import load_config

cfg = load_config()

def xml_to_dict(element):
    if element.tag == 'dict':
        res = {}
        it = iter(element)
        for child in it:
            if child.tag == 'key':
                key_name = child.text
                try:
                    value_node = next(it)
                    res[key_name] = xml_to_dict(value_node)
                except StopIteration:
                    break
        return res

    elif element.tag == 'array':
        return [xml_to_dict(child) for child in element]

    else:
        val = element.text if element.text else ""
        if element.tag == 'integer':
            return int(val) if val else 0
        elif element.tag == 'real':
            return float(val) if val else 0.0
        return val.strip()

def load_clean_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    main_dict = root.find('dict')
    if main_dict is not None:
        return xml_to_dict(main_dict)
    return {}

def get_patient_folders(root_dir):
    root_path = Path(root_dir)
    series_map = {}

    for patient_folder in root_path.iterdir():
        if patient_folder.is_dir():
            all_dcm_files = patient_folder.rglob("*.dcm")
            unique_series_folders = sorted(list(set(f.parent for f in all_dcm_files)))
            
            if unique_series_folders:
                series_map[patient_folder.name] = unique_series_folders
            else:
                print(f"Caution: {patient_folder.name} has no DICOM file.")
                
    return series_map

def get_patient_dicom_dict(root_dir):
    # Output dictionary: { PatientID: [ [file_path_1, file_path_2, ...], ... ] }
    patient_dicom_dict = defaultdict(list)
    folders_per_patient = get_patient_folders(root_dir)

    # 1. Iterate over each patient and their corresponding folders
    for pid, folder_list in folders_per_patient.items():
        patient_series_buffer = defaultdict(list)
        
        # Flag to ensure data integrity. 
        # If any file is corrupt or missing essential tags, discard the entire patient.
        is_valid_patient = True 

        # 2. Iterate over all folders belonging to the patient
        for folder_path in folder_list:
            if not is_valid_patient: 
                break # Stop processing if the patient is already marked invalid

            path = Path(folder_path)
            dcm_files = list(path.glob("*.dcm"))

            # 3. Iterate over DICOM files
            for fpath in dcm_files:
                try:
                    # Read header only to check metadata (Performance optimization)
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    
                    # Check for essential tags required for 3D reconstruction
                    if 'SeriesInstanceUID' not in ds or 'InstanceNumber' not in ds:
                        print(f"[Exclude] Skipping Patient {pid}: Missing essential tags in {fpath.name}")
                        is_valid_patient = False
                        break # Break file loop -> leads to breaking folder loop

                    series_uid = ds.SeriesInstanceUID
                    instance_num = int(ds.InstanceNumber)

                    # Buffer the data: Store as (InstanceNumber, FilePath) tuple for later sorting
                    patient_series_buffer[series_uid].append((instance_num, fpath))

                except Exception as e:
                    print(f"[Error] Skipping Patient {pid}: File read error in {fpath.name} - {e}")
                    is_valid_patient = False
                    break 
        
        # 4. Save data only if the patient is valid
        if is_valid_patient:
            for s_uid, file_tuples in patient_series_buffer.items():
                if not file_tuples: 
                    continue

                # Sort by InstanceNumber to ensure correct Z-axis ordering
                sorted_tuples = sorted(file_tuples, key=lambda x: x[0])
                
                series_files = []
                for _, fpath in sorted_tuples:
                    str_path = str(fpath).replace('\\', '/')
                    
                    # Extract relative path starting from 'patient' directory
                    # Regex logic: Remove everything before the first occurrence of 'patient'
                    extract_patient_path = re.sub(r'^.*?(?=patient)', '', str_path)
                    series_files.append(extract_patient_path)
                
                # Append the sorted series to the patient's record
                patient_dicom_dict[pid].append({s_uid: series_files})
                
        else:
            # Optional: Log excluded patients for debugging
            pass

    # Final verification
    print(f"Total valid patients processed: {len(patient_dicom_dict)}")

    return patient_dicom_dict

def extract_metadata_from_DICOM(row):
    """
    Extracts comprehensive metadata for Cardiac CT Segmentation and CAC Scoring.
    
    Target Metadata:
      1. Physics: Rescale Slope/Intercept, Mass Calibration Factor (for Agatston/Mass Score).
      2. Geometry: Pixel Spacing, Z-Spacing (Calculated), Origin (for XML matching).
      3. Protocol: Convolution Kernel, Cardiac Phase (for Error Analysis).
    
    Args:
        row (pd.Series): A row from the dataframe containing 'FilePaths'.
        
    Returns:
        pd.Series: Extracted metadata including Status, Geometry, and Physics parameters.
    """
    file_list = row['FolderList']
    
    # Validation: Return empty status if no files exist
    if not file_list: 
        return pd.Series({'Status': 'Empty'})

    try:
        # 1. Load Header Information
        # Read the first DICOM file to extract series-level metadata.
        # stop_before_pixels=True is used to minimize I/O overhead.
        dcm = pydicom.dcmread(os.path.join(cfg.paths.raw_data, file_list[0]), 
                              stop_before_pixels=True)
        
        # --- 2. Geometry Refinement (Z-Spacing Calculation) ---
        # DICOM 'SliceThickness' is often different from the actual spacing between slices.
        # We prioritize 'SpacingBetweenSlices' or calculate it directly from positions.
        slice_thick = float(getattr(dcm, 'SliceThickness', 0))
        spacing_tag = float(getattr(dcm, 'SpacingBetweenSlices', slice_thick))
        
        # Calculate actual Z-spacing using ImagePositionPatient difference (Ground Truth)
        if len(file_list) > 1:
            # Load the last slice to compute the total span
            last_dcm = pydicom.dcmread(os.path.join(cfg.paths.raw_data, file_list[-1]), stop_before_pixels=True)
            z_start = float(dcm.ImagePositionPatient[2])
            z_end = float(last_dcm.ImagePositionPatient[2])
            
            # Average spacing = Total Distance / (Number of Intervals)
            calculated_z_spacing = abs(z_end - z_start) / (len(file_list) - 1)
        else:
            # Fallback for single-slice series
            calculated_z_spacing = spacing_tag

        # --- 3. Cardiac Protocol & Quality Metrics ---
        # Extract Cardiac Phase (e.g., 75%) to identify motion artifacts (Systole vs Diastole)
        cardiac_phase = "Unknown"
        # Tag (0018,1082) Nominal Percentage of Cardiac Phase
        if (0x0018, 0x1082) in dcm:
            cardiac_phase = dcm[0x0018, 0x1082].value
        
        # Extract Convolution Kernel (e.g., 'I30f', 'B30f') for domain stratification.
        # Handle cases where the value is a list (MultiValue) or string.
        kernel = getattr(dcm, 'ConvolutionKernel', 'Unknown')
        if isinstance(kernel, (pydicom.multival.MultiValue, list)):
            kernel = str(list(kernel))

        # --- 4. Mass Scoring Factor Extraction ---
        # Tag (0018,9352): Calcium Scoring Mass Factor Device (Scanner specific calibration)
        # Usually returns a list of 3 floats; we use the mean value.
        mass_factors = dcm.get((0x0018, 0x9352), None)
        if mass_factors:
            mass_calib_factor = np.mean(mass_factors.value)
        else:
            # If missing, Mass Score cannot be calculated physically (set to 0.0)
            mass_calib_factor = 0.0

        # --- 5. Coordinate System for XML Matching ---
        # Extract Origin and Direction to map World Coordinates (mm) to Voxel Indices.
        origin = dcm.ImagePositionPatient # [x, y, z]
        direction = getattr(dcm, 'ImageOrientationPatient', [1,0,0,0,1,0])

        # --- 6. Pixel Representation Check ---
        # 0: Unsigned Integer, 1: Signed Integer (Crucial for correct HU conversion)
        pixel_rep = int(getattr(dcm, 'PixelRepresentation', 0))

        return pd.Series({
            'Status': 'OK',
            'NumSlices': len(file_list),
            
            # --- Physics Parameters (HU & Scoring) ---
            'RescaleSlope': float(getattr(dcm, 'RescaleSlope', 1)),
            'RescaleIntercept': float(getattr(dcm, 'RescaleIntercept', -1024)),
            'PixelRepresentation': pixel_rep,
            'MassCalibrationFactor': float(mass_calib_factor),
            
            # --- Geometry Parameters (Resampling) ---
            'PixelSpacing_X': float(dcm.PixelSpacing[1]), # Column spacing
            'PixelSpacing_Y': float(dcm.PixelSpacing[0]), # Row spacing
            'Z_Spacing': float(calculated_z_spacing),     # Calculated Z-spacing (Best accuracy)
            'SliceThickness': slice_thick,                # Reference for Partial Volume Effect
            
            # --- Coordinate System (XML Label Matching) ---
            'Origin_X': float(origin[0]), 
            'Origin_Y': float(origin[1]), 
            'Origin_Z': float(origin[2]), 
            'Direction': str(list(direction)),
            
            # --- Metadata for Analysis ---
            'ConvolutionKernel': str(kernel),
            'CardiacPhase': cardiac_phase,
            'Manufacturer': str(getattr(dcm, 'Manufacturer', 'Unknown')),
            'KVP': float(getattr(dcm, 'KVP', 120)),       # Tube Voltage (Dose info)
        })

    except Exception as e:
        # Error Handling: Log the error and mark status
        return pd.Series({'Status': f'Error: {str(e)}'})


def GeneratePatientData(root_dir, saveFileName=None):
    patient_dicom_dict = get_patient_dicom_dict(os.path.join(cfg.paths.raw_data, 'patient'))

    PATIENT_DATAFRAME = pd.DataFrame()
    for pid in patient_dicom_dict.keys():
        for patient_dict in patient_dicom_dict[pid]:
            SID = patient_dict.keys()
            folder_list = patient_dict.values()
            row = pd.DataFrame({"PID": [pid], "SID": SID, "FolderList": folder_list})
            PATIENT_DATAFRAME = pd.concat([PATIENT_DATAFRAME, row], ignore_index=True)

    PATIENT_DATAFRAME['PID'] = PATIENT_DATAFRAME['PID'].astype(int)
    PATIENT_DATAFRAME.sort_values(by='PID', inplace=True)
    PATIENT_DATAFRAME.reset_index(drop=True, inplace=True)

    meta_df = PATIENT_DATAFRAME.apply(extract_metadata_from_DICOM, axis=1)
    PATIENT_DATAFRAME = pd.merge(PATIENT_DATAFRAME, meta_df, left_index=True, right_index=True)

    if saveFileName:
        PATIENT_DATAFRAME.to_csv(os.path.join(cfg.paths.processed_data, saveFileName), index=False)
    return PATIENT_DATAFRAME