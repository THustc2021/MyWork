import os
import json
import matplotlib.pyplot as plt
import numpy as np

best_path = "/home/xtanghao/THPycharm/AEL_main/examples_th_new/unsupervised_TU_PROTEINS/ael_results_test/pops"

ls = [[] for _ in range(11)]
for name in os.listdir(best_path):
    if name.startswith("p"):
        with open(os.path.join(best_path, name), "r") as f:
            d = json.load(f)
        # ls[int(name[0])] = ls[int(name[0])] + (1 - d["objective"])
        for dd in d:
            ls[int(name.split("_")[-1].split(".")[0])].append(1 - dd["objective"])
y_mean = [np.mean(i) for i in ls]
y_best = [np.max(i) for i in ls]

x = list(range(11))

# 创建一个图形
plt.figure(figsize=(10, 6))

# 绘制曲线图
plt.plot(x, y_best, label='best', color='b', linewidth=5, linestyle='-', marker='o', markersize=24, alpha=0.5)
plt.plot(x, y_mean, label='mean', color='r', linewidth=5, linestyle='--', marker='s', markersize=24, alpha=0.5)

# 添加标题和标签
# plt.title('Accuracy during evolution in PROTEIN (3 pops, 3 algs)', fontsize=28)
plt.xlabel('Evolution Iterations', fontsize=20)
plt.ylabel('Fitness', fontsize=20)

# 网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(loc='lower right', fontsize=24)

plt.rcParams['xtick.labelsize'] = 24  # 设置X轴刻度标签大小
plt.rcParams['ytick.labelsize'] = 24  # 设置Y轴刻度标签大小

# 优化布局
plt.tight_layout()

# 显示图形
plt.show()