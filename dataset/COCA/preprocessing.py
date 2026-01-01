import os
import sys
from pathlib import Path
import json
import re
import argparse
import pandas as pd
import numpy as np
import pydicom
import plistlib
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm

# Configuration Settings
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

from utils.functions import load_config
from dataset.helper import get_patient_dicom_dict, extract_metadata_from_DICOM, get_image_and_mask


# Load config
cfg = load_config()

# Parser
def get_argparser():
    parser = argparse.ArgumentParser(description='Preprocessing COCA dataset')
    parser.add_argument('--root_dir', type=str, default=cfg.paths.raw_data)
    parser.add_argument('--saveFileName', type=str, help='Name of the output file ({dataset-name}_matched_data.csv)')
    return parser


# ==================================================================================
# [Expert Config] Load Class Mapping from External JSON
# This ensures code logic is decoupled from data definitions.
# ==================================================================================
def load_class_mapping(json_path):
    print(f">>> Loading Class Mapping from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check if it follows the structured format (parsing_map / display_map)
        if 'parsing_map' in data:
            return data['parsing_map']
        else:
            # Assume it's a simple flat dictionary
            return data
    except FileNotFoundError:
        print(f"❌ Error: Class mapping file not found at {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON from {json_path}")
        sys.exit(1)

# Load the mapping immediately using the config path
CLASS_MAPPING = load_class_mapping(cfg.data.class_mapping)

# ==================================================================================
# [Step 2 Helper] Robust Parsing Logic (With Filtering & Correction)
# ==================================================================================
def parse_and_save_annotation(pid, xml_path, save_dir, class_map):
    """
    Parses the Plist XML with strict filtering for empty ROIs.
    Uses the external class_map to assign IDs.
    """
    try:
        with open(xml_path, 'rb') as f:
            plist_data = plistlib.load(f)
    except Exception as e:
        print(f"[Error] Failed to load XML for PID {pid}: {e}")
        return None

    extracted_data = {}  # Key: SliceIndex(str), Value: List of ROIs
    found_classes = set()
    total_rois = 0

    # Data structure: Root -> Images (List)
    images = plist_data.get('Images', [])
    
    for img_entry in images:
        slice_idx = img_entry.get('ImageIndex')
        rois = img_entry.get('ROIs', [])
        
        if slice_idx is None or not rois:
            continue
            
        valid_rois = []
        for roi in rois:
            # --- [Crucial Check 1] Filter Empty ROIs (Ghost Data) ---
            # XML Data contains ROIs with 0 points or empty lists. Must be filtered.
            num_points = roi.get('NumberOfPoints', 0)
            points_px_raw = roi.get('Point_px', [])
            
            if num_points < 3 or not points_px_raw:
                continue

            # --- [Check 2] Name Mapping using External Config ---
            raw_name = roi.get('Name', '').strip()
            if raw_name not in class_map.keys() and raw_name != '':
                raw_name = 'Unknown'
            
            # Map string name to Integer ID using the loaded map
            class_id = class_map.get(raw_name)
            
            
            # if class_id == 'Unknown':
            #     # Optional: Log ignored classes if needed
            #     # print(f"Warning: Ignored class '{raw_name}' (not in map)")
            #     continue
                
            found_classes.add(raw_name)
            
            # --- [Check 3] Coordinate Parsing (Float String -> Int) ---
            parsed_polygon = []
            for pt_str in points_px_raw:
                # Format example: '(126.003922, 144.000000)'
                # Regex to extract numbers (including decimals)
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", pt_str)
                if len(matches) >= 2:
                    # Expert Tip: Coordinates in XML are floats.
                    # Must round and convert to Int for correct mask generation.
                    x = int(round(float(matches[0])))
                    y = int(round(float(matches[1])))
                    parsed_polygon.append([x, y])
            
            # Double check valid polygon formation (min 2 points)
            if len(parsed_polygon) > 1:
                valid_rois.append({
                    "class_id": class_id,
                    "class_name": raw_name, # Keep for debugging/verification
                    "points": parsed_polygon
                })
                total_rois += 1
        
        if valid_rois:
            extracted_data[str(slice_idx)] = valid_rois

    # Save logic (Intermediate JSON)
    if extracted_data:
        os.makedirs(save_dir, exist_ok=True)
        json_filename = f"{pid}.json"
        json_path = os.path.join(save_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=None)
            
        return {
            "HasAnnotation": True,
            "LabelPath": json_path,
            "TargetClasses": sorted(list(found_classes)),
            "NumAnnotatedSlices": len(extracted_data),
            "TotalROIs": total_rois
        }
    else:
        return {
            "HasAnnotation": False,
            "LabelPath": None,
            "TargetClasses": [],
            "NumAnnotatedSlices": 0,
            "TotalROIs": 0
        }

# ==================================================================================
# Main Pipeline
# ==================================================================================

def GeneratePatientData(root_dir):
    print(">>> [1/3] Loading DICOM Structure...")
    patient_dicom_dict = get_patient_dicom_dict(os.path.join(root_dir, 'patient'))

    PATIENT_DATAFRAME = pd.DataFrame()
    for pid in patient_dicom_dict.keys():
        for patient_dict in patient_dicom_dict[pid]:
             for sid, folder_path in patient_dict.items():
                row = pd.DataFrame({"PID": [int(pid)], "SID": [sid], "FolderList": [folder_path]})
                PATIENT_DATAFRAME = pd.concat([PATIENT_DATAFRAME, row], ignore_index=True)

    PATIENT_DATAFRAME.sort_values(by=['PID', 'SID'], inplace=True)
    PATIENT_DATAFRAME.reset_index(drop=True, inplace=True)

    print(">>> [2/3] Extracting Metadata...")
    meta_df = PATIENT_DATAFRAME.apply(extract_metadata_from_DICOM, axis=1)
    PATIENT_DATAFRAME = pd.merge(PATIENT_DATAFRAME, meta_df, left_index=True, right_index=True)

    return PATIENT_DATAFRAME

def main(args):
    # Step 1: Generate Metadata DataFrame
    patient_df = GeneratePatientData(args.root_dir)

    # Step 2: Process XML Annotations & Calculate Scores
    print("\n>>> [3/3] Processing XML Annotations & Calculating Scores...")
    
    XMLBASEPATH = os.path.join(cfg.paths.raw_data, 'calcium_xml')
    PROCESSED_LABEL_DIR = os.path.join(cfg.paths.processed_data, 'labels_json')
    
    # Initialize new columns
    patient_df['HasAnnotation'] = False
    patient_df['LabelPath'] = None
    patient_df['TargetClasses'] = None
    patient_df['Calculated_Score'] = 0.0  # <--- New Column for Agatston Score
    
    unique_pids = patient_df['PID'].unique()
    
    for pid in tqdm(unique_pids):
        # 1. Resolve XML Path
        xml_path = os.path.join(XMLBASEPATH, f"{pid}.xml")
        if not os.path.exists(xml_path):
             xml_path = os.path.join(XMLBASEPATH, f"{pid}.plist")

        if os.path.exists(xml_path):
            # 2. Parse & Save JSON (Using existing logic)
            meta_info = parse_and_save_annotation(
                pid, 
                xml_path, 
                PROCESSED_LABEL_DIR, 
                CLASS_MAPPING
            )
            
            if meta_info and meta_info['HasAnnotation']:
                # Update DataFrame flags
                mask_indices = patient_df['PID'] == pid
                patient_df.loc[mask_indices, 'HasAnnotation'] = True
                patient_df.loc[mask_indices, 'LabelPath'] = meta_info['LabelPath']
                
                # Store class information
                classes_str = str(meta_info['TargetClasses'])
                for idx in patient_df[mask_indices].index:
                    patient_df.at[idx, 'TargetClasses'] = classes_str
                
                # --- [NEW] Calculate Agatston Score ---
                # Call the scoring function
                score, _ = compute_agatston_score_and_mask(
                    json_path=meta_info['LabelPath'], 
                    dicom_folder=patient_df[patient_df['PID'] == pid]['FolderList'].values[0]
                )
                
                # Assign the calculated score to all rows for this PID
                patient_df.loc[mask_indices, 'Calculated_Score'] = score

    # Final Save
    save_path = os.path.join(cfg.paths.processed_data, args.saveFileName)
    patient_df.to_csv(save_path, index=False)
    
    print(f"\n✅ Pipeline Complete.")
    print(f"Saved to: {save_path}")
    
    # Check sample scores
    annotated_rows = patient_df[patient_df['HasAnnotation']]
    if not annotated_rows.empty:
        print("\n[Sample Agatston Scores]")
        print(annotated_rows[['PID', 'Calculated_Score']].drop_duplicates().head())


def get_dicom_file_map(dicom_folder):
    """
    Scans the DICOM folder to create a {InstanceNumber: FilePath} mapping.
    Purpose: To find the DICOM file corresponding to the XML ImageIndex in O(1) time.
    """
    mapping = {}

    if dicom_folder is None:
        return mapping

    for f in dicom_folder:
        if f.endswith('.dcm'):
            path = os.path.join(cfg.paths.raw_data, f)
            try:
                # Read header only (stop_before_pixels=True) for speed
                dcm = pydicom.dcmread(path, stop_before_pixels=True)
                idx = int(dcm.InstanceNumber)
                mapping[idx] = path
            except:
                continue
    return mapping

def compute_agatston_score_and_mask(json_path, dicom_folder, slice_offset=0):
    """
    Combines the saved JSON label and original DICOM to:
    1. Generate a 2D/3D Segmentation Mask (Using get_image_and_mask helper).
    2. Precisely calculate the Agatston Score (Calcium Score).
    
    Args:
        json_path (str): Path to the preprocessed JSON label file.
        dicom_folder (str): Path to the patient's DICOM folder.
        slice_offset (int): Offset between XML Index and DICOM Instance Number.
        
    Returns:
        total_score (float): The patient's total Agatston Score.
        mask_volume (dict): Mask data in format {slice_idx: 2d_numpy_array}.
    """
    
    # 1. Load JSON to get the list of annotated slices
    with open(json_path, 'r', encoding='utf-8') as f:
        roi_data = json.load(f)

    # 2. Map DICOM files for quick lookup
    dcm_map = get_dicom_file_map(dicom_folder)
    
    assert len(dcm_map) > 0, "No DICOM files found in the folder."
    
    total_score = 0.0
    mask_volume = {}
    
    # 3. Iterate only through slices with annotations
    for slice_idx_str in roi_data.keys():
        slice_idx = int(slice_idx_str)
        target_instance = slice_idx + slice_offset
        
        dcm_path = dcm_map.get(target_instance)
        if not dcm_path:
            continue

        # --- [Step A] Use Helper Function (Replacement) ---
        # This single line replaces DICOM loading, HU conversion, and Mask generation.
        try:
            image_hu, mask = get_image_and_mask(dcm_path, json_path, slice_offset)
        except Exception as e:
            print(f"Error processing slice {slice_idx}: {e}")
            continue

        mask_volume[slice_idx] = mask

        # --- [Step B] Get Metadata for Scoring (Extra Step) ---
        # Since get_image_and_mask returns numpy arrays only, 
        # we need to read the header again to get PixelSpacing.
        # 'stop_before_pixels=True' makes this very fast.
        dcm_meta = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        
        try:
            # Area = spacing_x * spacing_y
            area_per_pixel = float(dcm_meta.PixelSpacing[0]) * float(dcm_meta.PixelSpacing[1])
        except (AttributeError, TypeError):
            area_per_pixel = 0.5 * 0.5 # Default fallback

        # --- [Step C] Calculate Agatston Score (Same Logic) ---
        # 1. Filter: Intersection of Mask AND High Intensity (>130 HU)
        calcium_candidates = np.zeros_like(mask)
        calcium_candidates[(mask > 0) & (image_hu >= 130)] = 1
        
        # 2. Connected Component Analysis
        labeled_blobs = label(calcium_candidates) 
        blob_props = regionprops(labeled_blobs, intensity_image=image_hu)
        
        slice_score = 0.0
        
        for blob in blob_props:
            # Measure Max HU within the blob
            max_hu = blob.max_intensity
            
            # Apply Agatston Weighting Rule
            if max_hu >= 400: weight = 4
            elif max_hu >= 300: weight = 3
            elif max_hu >= 200: weight = 2
            elif max_hu >= 130: weight = 1
            else: weight = 0
            
            # Score = Area(mm2) * Weight
            slice_score += (blob.area * area_per_pixel * weight)
            
        total_score += slice_score

    return total_score, mask_volume


if __name__ == '__main__':
    arg = get_argparser()
    args = arg.parse_args()
    main(args)