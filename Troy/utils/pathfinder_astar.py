import heapq
import math

# 计算两点之间的曼哈顿距离
def manhattan_distance(start, end):
    return abs(end[0] - start[0]) + abs(end[1] - start[1])

# 寻路算法
def find_path(start_pos, end_pos, map_data):

    # 如果目标位置为障碍，直接返回None
    if not is_valid_tile(end_pos, map_data):
        return None

    # 定义移动方向（上、下、左、右以及对角线）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 创建起始节点
    start_node = (0, start_pos, None)

    # 初始化开放列表和关闭列表
    open_list = []
    closed_list = set()

    # 将起始节点加入开放列表
    heapq.heappush(open_list, start_node)

    # 开始寻路循环
    while open_list:
        print(open_list)
        # 从开放列表中取出F值最小的节点
        current_node = heapq.heappop(open_list)
        current_cost, current_pos, parent = current_node

        # 如果当前节点为目标节点，路径已找到，构造并返回路径
        if current_pos == end_pos:
            path = []
            while current_node:
                _, current_pos, parent = current_node
                path.append(current_pos)
                current_node = parent
            return path[::-1]

        # 将当前节点加入关闭列表
        closed_list.add(current_pos)

        # 遍历所有相邻节点
        for direction in directions:
            dx, dy = direction
            neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # 如果相邻节点不可通行或已在关闭列表中，则跳过
            if neighbor_pos in closed_list or not is_valid_tile(neighbor_pos, map_data):
                continue

            # 计算移动代价
            g_cost = current_cost + 1  # 在这里，每个相邻节点的移动代价都是1

            # 如果相邻节点不在开放列表中，或者通过当前节点到达相邻节点的移动代价更小
            if not is_in_open_list(neighbor_pos, open_list) or g_cost < get_cost(neighbor_pos, open_list):
                # 计算启发式评估函数（这里使用曼哈顿距离作为启发式评估）
                h_cost = manhattan_distance(neighbor_pos, end_pos)
                f_cost = g_cost + h_cost

                # 将节点加入开放列表
                heapq.heappush(open_list, (f_cost, neighbor_pos, current_node))

    # 如果开放列表为空，表示无法找到路径
    return None

# 检查坐标点是否在开放列表中
def is_in_open_list(pos, open_list):
    for _, node_pos, _ in open_list:
        if node_pos == pos:
            return True
    return False

# 获取坐标点在开放列表中的移动代价
def get_cost(pos, open_list):
    for cost, node_pos, _ in open_list:
        if node_pos == pos:
            return cost
    return math.inf

# 检查坐标点是否为可通行的地砖
def is_valid_tile(pos, map_data):
    if pos[1] >= map_data.shape[0] or pos[0] >= map_data.shape[1] or map_data[pos[1]][pos[0]] == 1:
        return False
    return True