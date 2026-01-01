import sys
from pathlib import Path
from box import ConfigBox

PROJ_ROOT = Path.cwd().parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

import xmltodict
import SimpleITK as sitk

from common import load_clean_xml


def load_single_xml(xml_path):
    """
    xml_path: xml file path
    return: List of RoI informations
    """
    return load_clean_xml(xml_path)['Images']

# class COCA_Dataset:
