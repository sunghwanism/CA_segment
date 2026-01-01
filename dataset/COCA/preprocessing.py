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
from tqdm import tqdm

# Configuration Settings
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

from utils.functions import load_config
from dataset.helper import get_patient_dicom_dict, extract_metadata_from_DICOM

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
            
            # Double check valid polygon formation (min 3 points)
            if len(parsed_polygon) > 2:
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
    # Step 1: DICOM DataFrame
    patient_df = GeneratePatientData(args.root_dir)

    # Step 2: Processing XML Annotations
    print("\n>>> [3/3] Processing XML Annotations...")
    print(f"🎯 Using Class Mapping from Config: {len(CLASS_MAPPING)} classes loaded.")
    
    XMLBASEPATH = os.path.join(cfg.paths.raw_data, 'calcium_xml')
    PROCESSED_LABEL_DIR = os.path.join(cfg.paths.processed_data, 'labels_json')
    
    patient_df['HasAnnotation'] = False
    patient_df['LabelPath'] = None
    patient_df['TargetClasses'] = None
    patient_df['NumAnnotatedSlices'] = 0
    
    unique_pids = patient_df['PID'].unique()
    
    for pid in tqdm(unique_pids):
        # 1. Resolve XML Path
        xml_path = os.path.join(XMLBASEPATH, f"{pid}.xml")
        if not os.path.exists(xml_path):
             xml_path = os.path.join(XMLBASEPATH, f"{pid}.plist")

        if os.path.exists(xml_path):
            # 2. Parse & Save JSON (Using loaded CLASS_MAPPING)
            meta_info = parse_and_save_annotation(
                pid, 
                xml_path, 
                PROCESSED_LABEL_DIR, 
                CLASS_MAPPING # Pass the config map here
            )
            
            if meta_info and meta_info['HasAnnotation']:
                # 3. Link to DataFrame
                mask = patient_df['PID'] == pid
                patient_df.loc[mask, 'HasAnnotation'] = True
                patient_df.loc[mask, 'LabelPath'] = meta_info['LabelPath']
                patient_df.loc[mask, 'NumAnnotatedSlices'] = meta_info['NumAnnotatedSlices']
                
                # Store class usage info
                classes_str = str(meta_info['TargetClasses'])
                for idx in patient_df[mask].index:
                    patient_df.at[idx, 'TargetClasses'] = classes_str

    # Final Save
    save_path = os.path.join(cfg.paths.processed_data, args.saveFileName)
    patient_df.to_csv(save_path, index=False)
    print(f"\n✅ Pipeline Complete. Saved to {save_path}")
    print(f"Total Annotated Entries: {patient_df['HasAnnotation'].sum()}")
    
    # Validation Hint
    print("\n[Action Required] Verify SID alignments using 'verify_dicom_xml_match' function.")

if __name__ == '__main__':
    arg = get_argparser()
    args = arg.parse_args()
    main(args)