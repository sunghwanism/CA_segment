import xml.etree.ElementTree as ET
from pathlib import Path
import json

import pydicom
from collections import defaultdict

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


def get_patient_series_map(root_dir):
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

def group_dicom_by_series(folder_path, pid):
    series_groups = defaultdict(list)
    path = Path(folder_path)
    
    dcm_files = list(path.glob("*.dcm"))
    
    for fpath in dcm_files:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            series_id = ds.SeriesInstanceUID 
            series_groups[series_id].append(fpath)
        except Exception as e:
            print(f"Fail to read file [pid: {pid}, file: {fpath.name}]: {e}")

    final_groups = {}
    for s_id, f_list in series_groups.items():
        sorted_slices = [pydicom.dcmread(f) for f in f_list]
        sorted_slices.sort(key=lambda x: int(x.InstanceNumber))
        
        desc = getattr(sorted_slices[0], 'SeriesDescription', s_id)
        final_groups[desc] = sorted_slices
        
    return final_groups

def load_all_patient_series(patient_id, series_map):
    if patient_id not in series_map:
        print(f"Caution: Patient {patient_id} not found.")
        return None
    
    series_folders = series_map[patient_id]
    all_series_data = {}
    
    for i, series_folder in enumerate(series_folders):
        files = list(series_folder.glob("*.dcm"))
        if not files: continue
        
        slices = [pydicom.dcmread(f) for f in files]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
        series_num = getattr(slices[0], 'SeriesNumber', i)
        all_series_data[f"Series_{series_num}"] = slices
        
    return all_series_data