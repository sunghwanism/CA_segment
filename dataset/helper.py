
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import json

import re
import ast
import cv2
import pandas as pd
import numpy as np
import pydicom
from collections import defaultdict, Counter

import plistlib
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


def verify_dicom_xml_match(df, row_index, dicom_root_dir, slice_offset=0):
    """
    Visualizes the overlay. Run this in Notebook or separate check script.
    """
    row = df.iloc[row_index]
    
    if not row['HasAnnotation'] or not row['LabelPath']:
        print(f"Row {row_index} (PID: {row['PID']}) has no annotation.")
        return

    with open(row['LabelPath'], 'r') as f:
        roi_data = json.load(f)
    
    annotated_indices = sorted([int(k) for k in roi_data.keys()])
    if not annotated_indices:
        print("Empty Annotation Data.")
        return

    target_instance_num = annotated_indices[len(annotated_indices)//2]
    series_path = os.path.join(dicom_root_dir, 'patient', str(row['FolderList']))
    
    dicom_file = None
    if os.path.exists(series_path):
        for f in os.listdir(series_path):
            if f.endswith('.dcm'):
                try:
                    dcm = pydicom.dcmread(os.path.join(series_path, f), stop_before_pixels=True)
                    if int(dcm.InstanceNumber) == (target_instance_num + slice_offset):
                        dicom_file = pydicom.dcmread(os.path.join(series_path, f))
                        break
                except:
                    continue
    
    if dicom_file is None:
        print(f"❌ Failed to find DICOM slice {target_instance_num + slice_offset}")
        return

    slope = float(getattr(dicom_file, 'RescaleSlope', 1))
    intercept = float(getattr(dicom_file, 'RescaleIntercept', 0))
    image = dicom_file.pixel_array.astype(np.float32) * slope + intercept
    
    center, width = 400, 1000
    img_min = center - width // 2
    img_max = center + width // 2
    image = np.clip(image, img_min, img_max)
    image = (image - img_min) / (img_max - img_min)

    H, W = image.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    slice_rois = roi_data.get(str(target_instance_num), [])
    
    print(f"🔍 Slice {target_instance_num}: {len(slice_rois)} ROIs found.")
    
    for roi in slice_rois:
        pts = np.array(roi['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Overlay PID:{row['PID']}")
    plt.imshow(image, cmap='gray')
    
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()


def scan_unique_roi_names(files, verbose=False):
    """
    Scans all XML/Plist files in the directory and counts the occurrences of each ROI Name.
    """
    
    # Counter to store Name: Count
    roi_name_counter = Counter()
    
    # Counter for errors
    error_count = 0
    
    for file_path in tqdm(files):
        
        try:
            with open(file_path, 'rb') as f:
                plist_data = plistlib.load(f)
            
            images = plist_data.get('Images', [])
            
            for img in images:
                rois = img.get('ROIs', [])
                if not rois:
                    print(f"No ROIs found in {file_path}")
                    continue
                
                for roi in rois:
                    # Extract Name
                    raw_name = roi.get('Name')
                    
                    if raw_name:
                        clean_name = raw_name.strip()
                        roi_name_counter[clean_name] += 1
                    else:
                        print(f"No Name Tag found in {file_path}")
                        roi_name_counter['(No Name Tag)'] += 1
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            error_count += 1
    
    if verbose:
        print("\n" + "="*50)
        print(f"✅ Scan Complete.")
        print(f"❌ Failed files: {error_count}")
        print("="*50)
    
    return roi_name_counter

def get_image_and_mask(dcm_filePath, LabelPath, slice_offset=0):
    """
    Loads a specific DICOM file and its corresponding annotation from a patient's JSON label file.
    Returns the Hounsfield Unit (HU) converted image and a binary segmentation mask.

    Args:
        dcm_filePath (str): Absolute path to the single .dcm file.
        LabelPath (str): Absolute path to the preprocessed .json label file for the patient.
        slice_offset (int): Offset to align DICOM InstanceNumber with XML ImageIndex.
                            (Default is 0, adjust if there is a mismatch).

    Returns:
        image_hu (np.array): The CT image converted to Hounsfield Units (dtype=np.float64).
        mask (np.array): Binary segmentation mask where 1 indicates the ROI (dtype=np.uint8).
    """
    
    # 1. Validation: Check if DICOM file exists
    if not os.path.exists(dcm_filePath):
        raise FileNotFoundError(f"DICOM file not found: {dcm_filePath}")
        
    # 2. Load DICOM file
    dcm = pydicom.dcmread(dcm_filePath)
    
    # 3. Convert raw pixel data to Hounsfield Units (HU)
    # Formula: HU = pixel_value * slope + intercept
    slope = getattr(dcm, 'RescaleSlope', 1.0)
    intercept = getattr(dcm, 'RescaleIntercept', 0.0)
    
    # Use float64 to prevent overflow and maintain precision during calculation
    image_hu = dcm.pixel_array.astype(np.float64) * slope + intercept
    
    # 4. Initialize an empty binary mask
    # The mask should have the same spatial dimensions (Height, Width) as the image
    H, W = image_hu.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 5. Validation: Check if Label file exists
    if not os.path.exists(LabelPath):
        # If no label file exists for this patient, return the empty mask
        # print(f"Warning: Label file not found at {LabelPath}. Returning empty mask.")
        return image_hu, mask
        
    # 6. Load JSON Label Data
    with open(LabelPath, 'r', encoding='utf-8') as f:
        roi_data = json.load(f)
        
    # 7. Match the DICOM Slice to the JSON Annotation
    # The 'InstanceNumber' tag in DICOM usually corresponds to the slice index.
    instance_num = int(dcm.InstanceNumber)
    
    # Apply offset if necessary (e.g., if XML index starts at 0 but DICOM starts at 1)
    target_key = str(instance_num - slice_offset)
    
    # 8. Draw ROIs on the Mask
    if target_key in roi_data:
        rois = roi_data[target_key]
        
        for roi in rois:
            # Extract points (list of [x, y]) and convert to numpy int32 array
            # OpenCV requires points to be int32
            pts = np.array(roi['points'], dtype=np.int32)
            
            # Fill the polygon on the mask
            # color=1: Assigns value 1 to the ROI area (Foreground)
            cv2.fillPoly(mask, [pts], color=1)
            
    return image_hu, mask