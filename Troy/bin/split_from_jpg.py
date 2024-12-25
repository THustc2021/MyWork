import os.path

import cv2

jpg_path = r"D:\Program-Station\Language-Python\Troy\assert\soldiers\short_sword_warrior_n\short_sword_warrior.png"
crop_positions = [  #x, y, w, h
    (83, 85, 132, 119),
    (230, 82, 149, 122),
    (396, 53, 137, 151),
    (536, 42, 125, 162),
    (701, 88, 103, 116),
    (83, 269, 132, 119),
    (235, 269, 132, 119),
    (379, 269, 148, 123),
    (566, 266, 106, 143),
    (678, 217, 133, 180)
]
save_path = r"D:\Program-Station\Language-Python\Troy\assert\soldiers\short_sword_warrior_n"
if not os.path.exists(save_path):
    os.makedirs(save_path)

img = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
# img = cv2.resize(img, (900, 450))
# cv2.imwrite(os.path.join(save_path, "ori.png"), img)
# # cv2.imshow("a", img)
# # cv2.waitKey(2000)
for i in range(len(crop_positions)):
    x, y, w, h= crop_positions[i]
    img_crop = img[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(save_path, f"{i}.png"), img_crop)