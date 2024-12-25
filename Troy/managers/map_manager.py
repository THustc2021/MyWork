import os
import pickle

import pytmx
import pyscroll.data
import numpy as np
from pytmx import load_pygame

from utils.pathfinder.graph.grid import GridMap
from utils.pathfinder.solver.jpsplus import JPSPlus
from utils.pathfinder.solver.pruning.base import BasePruning, NoPruning
from utils.pathfinder.solver.pruning.bbox import BBoxPruning
from utils.pathfinder.utils.distance import diagonalDistance

def get_map_size_in_px(filename):
    tmx_data = pytmx.TiledMap(filename)
    tw, th = tmx_data.tilewidth, tmx_data.tileheight
    return tw * tmx_data.width, th * tmx_data.height

class SimpleMap():

    def __init__(self, filename, solver_path=None):
        # 可通行map
        tmx_data = pytmx.TiledMap(filename)
        tw, th = tmx_data.tilewidth, tmx_data.tileheight
        access_map = np.zeros((tmx_data.width, tmx_data.height))
        for obj in tmx_data.objects:
            for i in range(int(obj.x), int(obj.x + obj.width)):
                for j in range(int(obj.y), int(obj.y + obj.height)):
                    access_map[int(i / tw)][int(j / th)] = 1  # 不可通行
        grid = GridMap()
        grid.readFromCells(access_map.shape[0], access_map.shape[1], access_map.tolist())   # 先宽后高
        # 加载地图解算器
        if solver_path is None:
            solver_path = filename.replace(".tmx", ".pickle")
        if not os.path.exists(solver_path):
            # prune = NoPruning()
            prune = BBoxPruning()
            solver = JPSPlus(diagonalDistance, prune)
            solver.doPreprocess(grid)
            with open(solver_path, "wb") as file:
                pickle.dump(solver, file)
        else:
            with open(solver_path, "rb") as file:
                solver = pickle.load(file)
        #
        self.tile_size = (tw, th)
        self.access_map = grid
        self.solver = solver
        self.world_size = (self.tile_size[0] * grid.width, self.tile_size[1] * grid.height)

class MainMap():
    """
    当窗口大小与地图大小不匹配时，需要此类进行控制。
    """

    def __init__(self, filename, screen_size, solver_path=None):

        # 创建渲染器
        tmx_data = load_pygame(filename)
        map_data = pyscroll.data.TiledMapData(tmx_data)
        self.map_layer = pyscroll.orthographic.BufferedRenderer(map_data, screen_size)
        # 可通行map
        tile_size = map_data.tile_size
        access_map = np.zeros((tmx_data.width, tmx_data.height))    # 先宽后高
        for obj in tmx_data.objects:    # 构建先宽后高的矩阵
            for i in range(int(obj.x), int(obj.x + obj.width)):
                for j in range(int(obj.y), int(obj.y + obj.height)):
                    access_map[int(i / tile_size[0])][int(j / tile_size[1])] = 1  # 不可通行
        grid = GridMap()
        grid.readFromCells(access_map.shape[0], access_map.shape[1], access_map.tolist()) # 这个access_map是转置的
        # 加载地图解算器
        if solver_path is None:
            solver_path = filename.replace(".tmx", ".pickle")
        if not os.path.exists(solver_path):
            # prune = NoPruning()
            prune = BBoxPruning()   # 算法中针对access_map的操作要注意是先x后y
            solver = JPSPlus(diagonalDistance, prune)
            solver.doPreprocess(grid)
            with open(solver_path, "wb") as file:
                pickle.dump(solver, file)
        else:
            with open(solver_path, "rb") as file:
                solver = pickle.load(file)
        #
        self.tile_size = tile_size
        self.access_map = grid
        self.solver = solver
        self.world_size = (self.tile_size[0] * grid.width, self.tile_size[1] * grid.height)

        # 设置摄像头初始视点
        self.center = [screen_size[0] / 2, screen_size[1] / 2]  # (x, y)

        # the camera vector is used to handle camera movement
        self.camera_acc = [0., 0.]    # 加速度
        self.camera_vel = [0., 0.]    # 速度

    def draw(self, surface):
        self.map_layer.draw(surface, surface.get_rect())

    def move_camera(self, direction, move_acc_speed=.2):
        # direction: 0 左， 1 上， 2 右， 3 下
        if direction == 0:
            self.camera_vel[1] = 0.
            self.camera_acc[1] = 0.
            if self.camera_vel[0] <= 0:  # 同向移动
                self.camera_acc[0] += move_acc_speed
                self.camera_vel[0] -= self.camera_acc[0]
            else:   # 暂停并反向
                self.camera_acc[0] = move_acc_speed
                self.camera_vel[0] = -move_acc_speed
        elif direction == 1:
            self.camera_vel[0] = 0.
            self.camera_acc[0] = 0.
            if self.camera_vel[1] <= 0: # 同向移动
                self.camera_acc[1] += move_acc_speed
                self.camera_vel[1] -= self.camera_acc[1]
            else:
                self.camera_acc[1] = move_acc_speed
                self.camera_vel[1] = -move_acc_speed
        elif direction == 2:
            self.camera_vel[1] = 0.
            self.camera_acc[1] = 0.
            if self.camera_vel[0] >= 0: # 同向移动
                self.camera_acc[0] += move_acc_speed
                self.camera_vel[0] += self.camera_acc[0]
            else:
                self.camera_acc[0] = move_acc_speed
                self.camera_vel[0] = move_acc_speed
        elif direction == 3:
            self.camera_vel[0] = 0.
            self.camera_acc[0] = 0.
            if self.camera_vel[1] >= 0.:
                self.camera_acc[1] += move_acc_speed
                self.camera_vel[1] += self.camera_acc[1]
            else:
                self.camera_acc[1] = move_acc_speed
                self.camera_vel[1] = move_acc_speed
        else:   # 全部重置
            self.camera_vel = [0, 0]
            self.camera_acc = [0, 0]

        self.center[0] += self.camera_vel[0]
        self.center[1] += self.camera_vel[1]

        # 出界计算
        view_size_w_half = self.map_layer.view_rect[2]/ 2
        view_size_h_half = self.map_layer.view_rect[3] / 2
        if self.center[0] - view_size_w_half < 0:
            self.center[0] = view_size_w_half
        elif self.center[0] + view_size_w_half >= self.world_size[0]:
            self.center[0] = self.world_size[0] - view_size_w_half
        if self.center[1] - view_size_h_half < 0:
            self.center[1] = view_size_h_half
        elif self.center[1] + view_size_h_half >= self.world_size[1]:
            self.center[1] = self.world_size[1] - view_size_h_half

        self.map_layer.center(self.center)