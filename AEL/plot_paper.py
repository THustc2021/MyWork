import json
import os

import matplotlib.pyplot as plt
import numpy as np

# 示例数据
# group_names = ['None', '+Aug.', '+lAug.', '+lAug.+Evo.', "+lAug.+Evo.+Eva."]
# group_names = ['None', 'Aug.', 'SimpleLLM', 'SimpleLLM+JOAOv2', "EvolutedLLM+JOAOv2"]
# categories = ['NCI1', 'COLLAB', 'DD', 'IMDB-BINARY', 'MUTAG', 'PROTEINS', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
# graph_data = [
#     [0.7693430656934306, 0.6297999999999999, 0.7716282775604809, 0.6689999999999999, 0.862280701754386, 0.745777027027027, 0.7085, 0.4112861723446894],
#     [0.7849959448499594, 0.717, 0.791749963783862, 0.7113333333333333, 0.8582846003898635, 0.7523273273273273, 0.9091666666666667, 0.5511786239144957],
#     [0.7914841849148418, 0.7152666666666666, 0.7555289970544207, 0.7013333333333334, 0.8763157894736843, 0.750849957099957, 0.9128333333333334, 0.557111957247829],
#     [0.7738037307380373, 0.7008, 0.7699406055338258, 0.7063333333333334, 0.8777777777777778, 0.7517186829686829, 0.8466666666666667, 0.5609106212424849],
#     [0.7708029197080293, 0.687, 0.776720266550775, 0.707, 0.8663742690058479, 0.7439591377091379, 0.8285, 0.5527130260521043],
# ]

group_names = ['None', 'SimpleLLM', "Ours"]
# categories = ['NCI1', 'COLLAB', 'DD', 'IMDB-BINARY', 'MUTAG', 'PROTEINS', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
# graph_data = [
#     [0.7693430656934306, 0.6297999999999999, 0.7716282775604809, 0.6689999999999999, 0.862280701754386, 0.745777027027027, 0.7085, 0.4112861723446894],
#     # [0.7728710462287105, 0.70065, 0.7688124728378966, 0.7128333333333333, 0.8697124756335284, 0.7488128753753754, 0.9019999999999999, 0.557111957247829],
#     [0.7788321167883212, 0.6916, 0.7648631029986962, 0.721, 0.8453216374269005,  0.7573359073359074, 0.8714999999999999, 0],
#     [0.7596107055961071, 0.7129, 0.7841, 0.6536666666666667, 0.8862573099415205, 0.7520457957957959, 0.8812, 0.5549],
# ]

categories = ['COLLAB', 'DD', 'MUTAG', 'PROTEINS']
graph_data = [
    [0.6297999999999999, 0.7716282775604809, 0.862280701754386, 0.745777027027027],
    [0.70065, 0.7688124728378966, 0.8697124756335284, 0.7488128753753754],
    # [0.7739659367396594, 0.6966, 0, 0, 0.8195906432748536, 0.7555743243243243, 0, 0],
    [0.7129, 0.7841, 0.8862573099415205, 0.7520457957957959],
]


# # 从文件中读取数据
# for name in categories:
#     name = "unsupervised_TU_" + name
#     # aug ----------------
#     aug_path = os.path.join("examples", name, "ael_seeds", "seeds.json")
#     with open(aug_path, "r") as f:
#         json_data = json.load(f)
#     # 计算平均值
#     augs = []
#     for data in json_data:
#         augs.append(1 - data["objective"])
#     graph_data[1].append(np.mean(augs))
#     # lAug -------------------
#     laugs = []
#     for sub_name in os.listdir(os.path.join("examples", name, "ael_results", "pops")):
#         with open(os.path.join("examples", name, "ael_results", "pops", sub_name), "r") as f:
#             laug_data = json.load(f)
#         for data in laug_data:
#             laugs.append(1 - data["objective"])
#     graph_data[2].append(np.mean(laugs))
#     # laug_evo --------------------
#     laugs_evo = []
#     for sub_name in os.listdir(os.path.join("examples", name, "ael_results", "history")):
#         with open(os.path.join("examples", name, "ael_results", "history", sub_name), "r") as f:
#             laug_data = json.load(f)
#         laugs_evo.append(1 - laug_data["offspring"]["objective"])
#     graph_data[3].append(np.mean(laugs_evo))
#     # all --------------
#     laugs_evo_eva = []
#     for sub_name in os.listdir(os.path.join("examples", name, "ael_results", "pops_best")):
#         with open(os.path.join("examples", name, "ael_results", "pops_best", sub_name), "r") as f:
#             laug_data = json.load(f)
#         laugs_evo_eva.append(1 - laug_data["objective"])
#     graph_data[4].append(np.mean(laugs_evo_eva))

# 颜色列表（浅色）
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightpink']

# 设置柱状图的宽度和组间的间距
bar_width = 0.15
index = np.arange(len(categories))

# 创建一个图形和一个子图
fig, ax = plt.subplots(figsize=(10, 6))

# 为每组数据绘制柱状图
for i in range(len(group_names)):
    plt.bar(index + i * bar_width, graph_data[i], bar_width, label=group_names[i], color=colors[i % len(colors)])

# 设置X轴标签和图例
plt.xlabel('Datasets', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(index + bar_width * 1.5, categories)

# 显示图例
plt.legend(fontsize=20)

# 美化网格线
plt.grid(True, linestyle='--', alpha=0.5)

plt.ylim(0.4, 0.95)
# 显示图形
plt.tight_layout()
plt.show()