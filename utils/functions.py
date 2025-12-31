import sys
import yaml
from box import ConfigBox
import os

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))


def load_config():
    yaml_path = os.path.join(PROJ_ROOT, 'config', 'config.yaml')

    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = ConfigBox(yaml.safe_load(f))
    
    return cfg