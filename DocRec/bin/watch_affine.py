import cv2
import matplotlib.pyplot as plt
import numpy as np

# 加载原始图像
image = cv2.imread("/home/xtanghao/THPycharm/dataset/DocReal/scanned/1.png")
image_height, image_width = image.shape[:2]

# 定义投影变换的四个顶点
source_points = np.float32([[0, 0], [image_width - 1, 0], [image_width - 1, image_height - 1], [0, image_height - 1]])

# 定义目标图像的四个顶点（可以根据需要进行调整）
target_points = np.float32([[0, 0], [image_width - 1, 0], [image_width - 1, image_height - 1], [0, image_height - 1]])

# 计算投影变换矩阵
transform_matrix = cv2.getPerspectiveTransform(source_points, target_points)

# 应用投影变换
transformed_image = cv2.warpPerspective(image, transform_matrix, (image_width, image_height))

# 显示原始图像和变换后的图像
# cv2.imshow("Original Image", image)
# cv2.imshow("Transformed Image", transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(image)
plt.show()
plt.imshow(transformed_image)
plt.show()