import os
import json
import matplotlib.pyplot as plt
import numpy as np

best_path = "/home/xtanghao/THPycharm/AEL_main/examples/unsupervised_TU_PROTEINS/ael_results/pops_best"

ls = [[] for _ in range(10)]
for name in os.listdir(best_path):
    if not name.startswith("p"):
        with open(os.path.join(best_path, name), "r") as f:
            d = json.load(f)
        # ls[int(name[0])] = ls[int(name[0])] + (1 - d["objective"])
        ls[int(name[0])].append(1 - d["objective"])
y_mean = [np.mean(i) for i in ls]
y_best = [np.max(i) for i in ls]

x = list(range(1, 11))

# 创建一个图形
plt.figure(figsize=(10, 6))

# 绘制曲线图
plt.plot(x, y_best, label='best', color='b', linewidth=5, linestyle='-', marker='o', markersize=6)
plt.plot(x, y_mean, label='mean', color='r', linewidth=5, linestyle='--', marker='s', markersize=6)

# 添加标题和标签
plt.title('Accuracy during evolution in PROTEIN (3 pops, 3 algs)', fontsize=16)
plt.xlabel('Evolution Iterations', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

# 网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(loc='upper left', fontsize=12)

# 优化布局
plt.tight_layout()

# 显示图形
plt.show()