import xml.etree.ElementTree as ET
import os
import csv

def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax, name])
    return boxes

def convert_to_yolo_format(xml_dir, output_dir, img_width, img_height):
    os.makedirs(output_dir, exist_ok=True)
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            boxes = parse_annotation(xml_path)
            txt_file = xml_file.replace('.xml', '.txt')
            with open(os.path.join(output_dir, txt_file), 'w') as f:
                for box in boxes:
                    xmin, ymin, xmax, ymax, cls = box
                    class_id = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrectly': 2}[cls]
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")