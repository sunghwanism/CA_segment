import os
import sys

import yaml
from box import ConfigBox
import plistlib


PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))


def load_config():
    yaml_path = os.path.join(PROJ_ROOT, 'config', 'config.yaml')

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
                print(f"{spacing}  - {key}: <{type(value).__name__}>")
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


