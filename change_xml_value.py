import xml.etree.ElementTree as ET
import os
import re


# DIRECTORY = "C:\\GitHub\\dds_drone_ws\\scripts\\preprocessing\\"
# DIRECTORY = "C:\\DDS_DATA_SET\\20230703\\notdrone\\"
DIRECTORY = "C:\\DDS_DATA_SET\\20230703\\drone\\"

ELEMENT = 'object/name'
# NEW_VALUE = 'notdrone'
NEW_VALUE = 'drone'

xml_files = [f for f in os.listdir(DIRECTORY)
            if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.xml)$', f)]

for image_xml in xml_files:
    fullpath = DIRECTORY  + image_xml
    tree = ET.parse(fullpath)
    root = tree.getroot()
    element_to_change = root.findall(ELEMENT)
    for el in element_to_change:
        print(image_xml,el.text)
        el.text = NEW_VALUE
    tree.write(fullpath)

