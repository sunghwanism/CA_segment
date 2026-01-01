import os
import sys
from pathlib import Path

import yaml
from box import ConfigBox
import plistlib


def load_config():
    PROJ_ROOT = get_project_root()
    yaml_path = os.path.join(PROJ_ROOT, 'config', 'config.yaml')

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found at {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = ConfigBox(yaml.safe_load(f))
    
    return cfg


def get_xml_schema(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    schema_map = {}

    for elem in tree.iter():
        if elem.tag not in schema_map:
            schema_map[elem.tag] = set()
        
        schema_map[elem.tag].update(elem.attrib.keys())
    
    print(f"=== XML Schema Summary: {os.path.basename(xml_path)} ===\n")
    for tag, attrs in schema_map.items():
        attr_list = ", ".join(sorted(attrs))
        if not attr_list:
            attr_list = "(No Attributes)"
        print(f"Tag: <{tag}>\n  └── Attributes: [{attr_list}]")


def get_plist_schema(data, indent=0, parent_key="Root"):
    """
    Recursively inspect dictionary structure to print schema (Keys & Types).
    Skips printing full content, focuses on structure.
    """
    spacing = "  " * indent
    
    # 1. Dictionary Type
    if isinstance(data, dict):
        print(f"{spacing}📂 [Dict] (Parent: {parent_key})")
        for key, value in data.items():
            if not isinstance(value, (dict, list)):
                # Print Type and a small preview of the value for primitive types
                val_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"{spacing}  - {key}: <{type(value).__name__}> (Sample: {val_preview})")
            else:
                print(f"{spacing}  - {key}: ", end="")
                get_plist_schema(value, indent + 1, parent_key=key)

    # 2. List Type (Array)
    elif isinstance(data, list):
        print(f"\n{spacing}📜 [List] Length: {len(data)}")
        if len(data) > 0:
            print(f"{spacing}  (Schema of first element ONLY):")
            get_plist_schema(data[0], indent + 1, parent_key=f"{parent_key}[0]")
        else:
            print(f"{spacing}  (Empty List)")

    # 3. Primitive Type (Fallback)
    else:
        print(f"<{type(data).__name__}>")


def inspect_xml_file(file_path):
    print(f"🔍 Inspecting file: {file_path}")
    print("="*60)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    try:
        # Expert Tip: Open in binary mode ('rb') because plistlib handles 
        # both XML-based and Binary-based Plist files automatically.
        with open(file_path, 'rb') as f:
            data = plistlib.load(f)
            
        # Run the schema printer
        get_plist_schema(data)
        
    except Exception as e:
        print(f"❌ Failed to parse Plist: {e}")
        print("Tip: Check if the file is a valid XML/Plist format.")

def get_project_root():

    try:
        current_path = Path(__file__).resolve()
    except NameError:
        current_path = Path.cwd().resolve()

    root = current_path
    Found = False

    while not Found:
        if os.path.exists(os.path.join(root, 'setup.py')):
            Found = True
        else:
            root = root.parent

    return root