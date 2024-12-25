from typing import SupportsFloat, Any

import numpy as np
import pygame

import gymnasium as gym
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete

from managers.display_manager import MainGroup
from managers.map_manager import MainMap, SimpleMap
from units import troy_soldiers, greece_soldiers, soldier


class Player():

    def __init__(self, camp):

        self.camp = camp   # 0 or 1

class TroyEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps":30}

    def __init__(self, render_mode=None, win_width=1024, win_height=640):

        self.agents = [Player(0), Player(1)]    # 0为troy，1为greece
        self.window_size = (win_width, win_height)

        # 观察环境主要为各个单位的位置，各项属性及状态
        troy_num = 23
        greece_num = 23
        self.observation_space = spaces.Dict({
            # 在游戏中，移动状态，逃跑状态
            "troy_state": Box(low=np.array([[0, 0, 0]]).repeat(troy_num, axis=0), high=np.array([[1, 2, 1]]).repeat(troy_num, axis=0), shape=(troy_num, 3), dtype=int),
            # 士气，人数，体力，近战攻击，近战防御，冲锋加成，远程防御
            "troy_attr": Box(low=0, high=np.array([[100, 1000, 100, 200, 200, 200, 200]]).repeat(troy_num, axis=0), shape=(troy_num, 7), dtype=float),
            "greek_state": Box(low=np.array([[0, 0, 0]]).repeat(greece_num, axis=0), high=np.array([[1, 2, 1]]).repeat(greece_num, axis=0), shape=(greece_num, 3), dtype=int),
            "greek_attr": Box(low=0, high=np.array([[100, 1000, 100, 200, 200, 200, 200]]).repeat(greece_num, axis=0), shape=(greece_num, 7), dtype=float)   # 各项属性标志
        })
        self.action_space = spaces.Dict({
            # 脱离战斗，技能1，技能2，技能q，移动，追击
            "troy_command": Box(low=0, high=1, shape=(troy_num, 6), dtype=int),
            # 设置移动方位（地图大小）
            "troy_move": Box(low=0, high=np.array([[32*50-1, 32*50-1]]).repeat(troy_num, axis=0), shape=(troy_num, 2), dtype=int),
            # 设置追击敌军
            "troy_chase": Box(low=0, high=greece_num-1, shape=(troy_num, ), dtype=int),
            "greek_command": Box(low=0, high=1, shape=(greece_num, 6), dtype=int),
            # 设置移动方位
            "greek_move": Box(low=0, high=np.array([[32*50-1, 32*50-1]]).repeat(greece_num, axis=0), shape=(greece_num, 2), dtype=int),
            # 设置追击敌军
            "greek_chase": Box(low=0, high=troy_num-1, shape=(greece_num, ), dtype=int)
        })

        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.gameMap = None
        self.gameGroup = None
        self.troy_troop = []
        self.greece_troop = []

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if self.render_mode == "human":
            import os
            os.environ['SDL_VIDEODRIVER'] = 'windows'
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 200)
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Troy')
            pygame.display.set_icon(pygame.image.load("assert/icon.png"))
            gameMap = MainMap('assert/maps/grasslands/grasslands.tmx', self.window_size)
            gameGroup = MainGroup(gameMap.map_layer, self.window_size)
            self.clock = pygame.time.Clock()
        else:
            gameMap = SimpleMap('assert/maps/grasslands/grasslands.tmx')
            gameGroup = MainGroup()

        # 初始化环境
        t0 = troy_soldiers.Infrantry((300, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t1 = troy_soldiers.Infrantry((360, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t2 = troy_soldiers.Infrantry((420, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t3 = troy_soldiers.Infrantry((480, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t4 = troy_soldiers.Infrantry((540, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t5 = troy_soldiers.Infrantry((600, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t6 = troy_soldiers.Infrantry((660, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t7 = troy_soldiers.Infrantry((720, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t8 = troy_soldiers.Rider((800, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t9 = troy_soldiers.Rider((860, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t10 =troy_soldiers.Rider((920, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t11 = troy_soldiers.Rider((980, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t12 = troy_soldiers.Archer((160, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t13 = troy_soldiers.Archer((220, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t14 = troy_soldiers.Archer((280, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t15 = troy_soldiers.Archer((340, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t16 = troy_soldiers.Archer((400, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t17 = troy_soldiers.Archer((460, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t18 = troy_soldiers.Archer((520, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t19 = troy_soldiers.TroyGuard((260, 500), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t20 = troy_soldiers.TroyGuard((400, 500), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t21 = troy_soldiers.Hector((340, 500), gameGroup=gameGroup, gameMap=gameMap, which_troop=0,
                             is_hero=True)
        t22 = troy_soldiers.Priam((150, 200), gameGroup=gameGroup, gameMap=gameMap, which_troop=0,
                            is_king=True)

        g0 = greece_soldiers.GreeceArcher((800, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g1 = greece_soldiers.GreeceArcher((860, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g2 = greece_soldiers.GreeceArcher((920, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g3 = greece_soldiers.GreeceArcher((980, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g4 = greece_soldiers.AthensMan((1040, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                  which_troop=1)
        g5 = greece_soldiers.AthensMan((1100, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                  which_troop=1)
        g6 = greece_soldiers.MopoliaArcher((1200, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                      which_troop=1)
        g7 = greece_soldiers.MopoliaArcher((1260, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                      which_troop=1)
        g8 = greece_soldiers.MycenaeanSworder((800, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g9 = greece_soldiers.MycenaeanSworder((860, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g10 = greece_soldiers.MycenaeanSworder((920, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g11 = greece_soldiers.MycenaeanSworder((980, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g12 = greece_soldiers.MycenaeanSworder((1040, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g13 = greece_soldiers.MycenaeanSworder((1100, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g14 = greece_soldiers.MycenaeanSworder((1160, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g15 = greece_soldiers.ArgosMan((860, 1400), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
        g16 = greece_soldiers.ArgosMan((920, 1400), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
        g17 = greece_soldiers.PylosMan((720, 1300), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
        g18 = greece_soldiers.PylosMan((1240, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                 which_troop=1)
        g19 = greece_soldiers.SpartarMan((980, 1400), gameGroup=gameGroup, gameMap=gameMap,
                                   which_troop=1)
        g20 = greece_soldiers.SpartarMan((1040, 1400), gameGroup=gameGroup, gameMap=gameMap,
                                   which_troop=1)
        g21 = greece_soldiers.Achilles((1000, 1500), gameGroup=gameGroup, gameMap=gameMap,
                                 which_troop=1, is_hero=True)
        g22 = greece_soldiers.AkaMenon((1080, 1500), gameGroup=gameGroup, gameMap=gameMap,
                                 which_troop=1, is_king=True)

        self.gameMap = gameMap
        self.gameGroup = gameGroup
        self.troy_troop = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10,
                            t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
                            t21, t22]
        self.greece_troop = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10,
                          g11, g12, g13, g14, g15, g16, g17, g18, g19, g20,
                          g21, g22]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        troy_state = []
        troy_attr = []
        for t in self.troy_troop:
            troy_state.append(np.array([t.out_of_game, t.current_moving_state, t.current_flee_state]))
            troy_attr.append(np.array([t.current_morale, t.current_soldier_nums, t.current_strength,
                                       t.current_close_combat_attack, t.current_close_combat_defense,
                                       t.current_charge_add, t.current_remote_combat_defense]))
        greek_state = []
        greek_attr = []
        for t in self.greece_troop:
            greek_state.append(np.array([t.out_of_game, t.current_moving_state, t.current_flee_state]))
            greek_attr.append(np.array([t.current_morale, t.current_soldier_nums, t.current_strength,
                                       t.current_close_combat_attack, t.current_close_combat_defense,
                                       t.current_charge_add, t.current_remote_combat_defense]))
        return {"troy_state": np.array(troy_state), "troy_attr": np.array(troy_attr),
                "greek_state": np.array(greek_state), "greek_attr": np.array(greek_attr)}
    def _get_info(self):
        return {}
    def _render_frame(self):
        # 显示模式
        if self.render_mode == "human":
            # 画面更新
            self.gameGroup.draw(self.window)
            pygame.display.flip()
            pygame.event.pump()
            # 允许交互
            event_handle(self.gameGroup, self.gameMap)
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            return None

    def step(
        self, action: ActType
    ) :
        ## 执行所有玩家的命令
        troy_command = action["troy_command"]
        troy_move = action["troy_move"]
        troy_chase = action["troy_chase"]
        for i in range(len(self.troy_troop)):
            s = self.troy_troop[i]
            if s.out_of_game:
                continue
            if troy_command[i][0]:
                if s.in_close_combat():
                    s.command_disengage_combat()
            if troy_command[i][1]:
                s.command_ability1()
            if troy_command[i][2]:
                s.command_ability2()
            if troy_command[i][3]:
                s.command_abilityq()
            if troy_command[i][5]:  # 追击优先级>移动
                s.command_chasing_enemy(self.greece_troop[troy_chase[i]])
            elif troy_command[i][4]:
                s.command_moving_position(troy_move[i])
        greek_command = action["greek_command"]
        greek_move = action["greek_move"]
        greek_chase = action["greek_chase"]
        for j in range(len(self.greece_troop)):
            s = self.greece_troop[j]
            if s.out_of_game:
                continue
            if greek_command[j][0]:
                if s.in_close_combat():
                    s.command_disengage_combat()
            if greek_command[j][1]:
                s.command_ability1()
            if greek_command[j][2]:
                s.command_ability2()
            if greek_command[j][3]:
                s.command_abilityq()
            if greek_command[j][5]:  # 追击优先级>移动
                s.command_chasing_enemy(self.troy_troop[greek_chase[j]])
            elif greek_command[j][4]:
                s.command_moving_position(greek_move[j].tolist())
        # 更新状态与跳出判断
        self.gameGroup.update()

        reward = [-np.sum(troy_command), -np.sum(greek_command)]    # 尽可能地保持更少的命令数量
        observation = self._get_obs()
        if self.gameGroup.soldier_manager.check_player_troop_destroyed():
            terminated = True
            reward = [np.inf, -np.inf]
            print("troy win!")
        elif self.gameGroup.soldier_manager.check_enemy_troop_destroyed():
            terminated = True
            reward = [-np.inf, np.inf]
            print("greek win!")
        else:
            terminated = False
        info = self._get_info()

        # 渲染界面
        self._render_frame()

        return observation, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

from pygame.locals import *
def event_handle(gameGroup, gameMap):

    for event in pygame.event.get():

        gccs = gameGroup.current_choose_s
        if event.type == QUIT:
            return False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
            if gccs != None:
                if event.key == K_e: # stop moving
                    gccs.command_moving_state(0)
                elif event.key == K_r:
                    gccs.command_moving_state(1)
                elif event.key == K_t:
                    gccs.command_moving_state(2)
                elif event.key == K_1:
                    gccs.command_ability1()
                elif event.key == K_2:
                    gccs.command_ability2()
                elif event.key == K_q:
                    gccs.command_abilityq()
        elif event.type == MOUSEBUTTONDOWN:
            choose_flag = False  # 本次点击是否选中了精灵
            mouse_pos = pygame.mouse.get_pos()
            mouse_world_pos = gameGroup.screen_to_world(*mouse_pos)
            if event.button == 1:  # 左键点击
                # 遍历组中的精灵，判断是否有精灵被点击，点击位置离谁的中心最近选的就是谁
                min_dist = -1
                choosed_spr = None
                for sprite in gameGroup:
                    if isinstance(sprite, soldier.Soldier) and  \
                            sprite.rect.collidepoint(mouse_world_pos) \
                            and not sprite.is_surrending():  # 通过此法可以定位精灵
                        # 计算距离
                        c = sprite.rect.center
                        dist = (mouse_world_pos[0] - c[0]) ** 2 + (mouse_world_pos[1] - c[1]) ** 2
                        if min_dist == -1 or dist < min_dist:
                            choosed_spr = sprite
                # 设置选中
                if choosed_spr != None:
                    choose_flag = True
                    gameGroup.set_choose(choosed_spr)
            if not choose_flag:  # 取消选择
                gameGroup.set_choose(None)
        elif event.type == MOUSEWHEEL: # 向上滚轮
            if event.y > 0:
                gameMap.map_layer.zoom += 0.25
            else:
                value = gameMap.map_layer.zoom - 0.25
                if value > 0:
                    gameMap.map_layer.zoom = value

    pressed = pygame.key.get_pressed()
    # 摄像头移动事件
    if pressed[pygame.K_a]:
        gameMap.move_camera(0)
    elif pressed[pygame.K_w]:
        gameMap.move_camera(1)
    elif pressed[pygame.K_d]:
        gameMap.move_camera(2)
    elif pressed[pygame.K_s]:
        gameMap.move_camera(3)
    else:
        gameMap.move_camera(-1)

    return True