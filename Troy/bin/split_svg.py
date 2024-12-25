import os.path
import xml.etree.ElementTree as ET
import base64
import cv2
import numpy as np

tree = ET.parse(r'D:\Program-Station\Language-Python\Troy\assert\2204_w053_n004_15_medicharacters_p1_15-ai.svg')
save_dir = os.path.join(r"D:\Program-Station\Language-Python\Troy\assert\soldiers", "short_sword_warrior")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

idx = 0
for element in tree.iter():
    if element.tag == '{http://www.w3.org/2000/svg}defs':
        for e in element.iter():
            if e.get("href") != None:
                img = base64.b64decode(e.attrib['href'].split(",")[1])
                # 二进制数据流转np.ndarray [np.uint8: 8位像素]
                img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(save_dir, f"{idx}.png"), img)

                idx += 1