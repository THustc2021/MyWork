import random
import pygame
import numpy as np

import struct

def unsigned_to_signed_32_bit(num):
    if num < 0: # 不处理有符号数
        return num
    # 使用'I'格式代码将无符号整数转换为4字节（32位）无符号整数
    packed = struct.pack('I', num)
    # 使用'i'格式代码将字节数据解包为带符号整数
    unpacked = struct.unpack('i', packed)[0]
    return unpacked

def get_diff_color_level(percent, level_list=(0.75, 0.5, 0.25)):

    if percent > level_list[0]:
        return (0, 255, 0)
    elif percent > level_list[1]:
        return (255, 255, 0)
    elif percent > level_list[2]:
        return (255, 0, 0)
    else:
        return (255, 255, 255)

def get_diff_level(percent, level_list=(0.75, 0.5, 0.25, 0.01), words=("士气高昂", "自信的", "动摇", "士气崩溃", "溃不成军")):
    for lvli in range(len(level_list)):
        if percent > level_list[lvli]:
            return words[lvli]
    return words[-1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def set_in_range(value, lower_bound=None, up_bound=None):
    if lower_bound != None:
        if value < lower_bound:
            return lower_bound
    if up_bound != None:
        if value > up_bound:
            return up_bound
    return value

def array_distance(arr1, arr2):
    '''
    计算两个数组里，每任意两个点之间的L2距离
    arr1 和 arr2 都必须是numpy数组
    且维度分别为 m x 2, n x 2
    输出数组的维度为 m x n
    '''
    if len(arr1.shape) != 2 or len(arr2.shape) != 2:
        return np.zeros(1)
    m, _ = arr1.shape
    n, _ = arr2.shape
    arr1_power = np.power(arr1, 2)
    arr1_power_sum = arr1_power[:, 0] + arr1_power[:, 1]
    arr1_power_sum = np.tile(arr1_power_sum, (n, 1))
    arr1_power_sum = arr1_power_sum.T
    arr2_power = np.power(arr2, 2)
    arr2_power_sum = arr2_power[:, 0] + arr2_power[:, 1]
    arr2_power_sum = np.tile(arr2_power_sum, (m, 1))
    dis = arr1_power_sum + arr2_power_sum - (2 * np.dot(arr1, arr2.T))
    dis = np.sqrt(dis)
    return dis

def direction_check(v, nvs):
    # 当前向量与向量集中所有向量的夹角是否都非锐角
    # 二元方向向量（单位）
    nvs = nvs.T
    ans = v @ nvs
    return np.any(ans>0)   # 若至少有一个为锐角，返回True，否则False

def cal_distance(wps1, wps2):
    return ((wps1[0] - wps2[0])**2 + (wps1[1] - wps2[1]) ** 2) **(1/2)

def sign2dvect(v):
    return (
        v[0] // abs(v[0]) if v[0] != 0 else 0,
        v[1] // abs(v[1]) if v[1] != 0 else 0)

def spiral_search(t):    # 螺旋探索周围的点
    # step是当前这一趟走的次数（一个step要往两个方向走两趟，用flag标识）
    if t <= 1:
        return 1, 0, 1, 0, 1, 1, 0

    x, y, last_move, x_direct, y_direct, step, flag = spiral_search(t-1)

    x = x + x_direct
    y = y + y_direct
    last_move -= 1
    # 获得新的last_move
    if last_move == 0:
        nx_direct = x_direct - y_direct if x_direct == 0 else 0
        ny_direct = x_direct if y_direct == 0 else 0
        if flag == 0:
            step += 1
            flag = 1
        else:
            flag = 0
        last_move = step
    else:
        nx_direct = x_direct
        ny_direct = y_direct

    return x, y,  last_move, nx_direct, ny_direct, step, flag