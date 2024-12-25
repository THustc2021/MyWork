import functools

import numpy as np
import pygame

from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from managers.display_manager import MainGroup
from managers.map_manager import MainMap, SimpleMap, get_map_size_in_px
from units import troy_soldiers, greece_soldiers, soldier
from units.remote_soldier import RemoteSoldier
from utils.common import array_distance

NUM_ITERS = 3000
class TroyEnv(ParallelEnv):

    metadata = {"render_modes": ["human"], "name": {"troy_v0"}, "render_fps":30}

    def __init__(self, render_mode=None, win_width=1024, win_height=640, event_handler=None, map_path="assert/maps/grasslands/grasslands_mid_mod.tmx"):

        self.possible_agents = ["player_" + str(r) for r in range(2)]
        # 记录每个势力对应的己方和敌方单位初始数量
        self.agent_name_mapping = dict(
            zip(self.possible_agents, [(23, 23), (23, 23)])
        )

        self.render_mode = render_mode

        self.window = None
        self.window_size = (win_width, win_height)
        self.clock = None
        self.gameMap = None
        self.gameGroup = None
        self.troy_troop = []
        self.greek_troop = []
        self.map_path = map_path

        # 加载地图，地图大小会影响到观测空间的上限
        self.world_size = get_map_size_in_px(map_path)

        self.event_handler = event_handler

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        num = self.agent_name_mapping[agent][0]
        enum = self.agent_name_mapping[agent][1]
        # 我们不单独设离开游戏观察，当离开游戏时，所有变量都置-1
        # 移动状态，逃跑中，有效战斗，逃离战斗，不受限移动，被围攻，被夹击，处在实际战斗中
        # 士气，人数，体力，近战攻击，近战防御，冲锋加成，远程防御，世界坐标，在远程作战中，远程弹药量
        # 当前游戏时间
        # 组合起来更加方便处理，前面为状态，后面为属性
        # 己方状态和部分敌方状态
        return spaces.Dict({
            "mask": Box(0, 1, (num, 6), dtype=np.int32),
            "obs_time": Box(low=0, high=NUM_ITERS, shape=(1,)),
            "obs_self": Box(low=np.ones((num, 20), dtype=np.float32)*-1,
                   high=np.array([[3, 2, 1, 1, 1, 1, 1, 1, 1,
                                   20, 1000, 20, 200, 200, 200, 200,
                                   self.world_size[0] - 1, self.world_size[1] - 1, 1, 200]], np.float32).repeat(num, axis=0),
                   shape=(num, 20), dtype=np.float32),
            "obs_enemy": Box(low=np.ones((enum, 13), dtype=np.float32)*-1,
                             high=np.array([[3, 2, 1, 1, 1, 1, 1, 1, 1, 1000, self.world_size[0] - 1, self.world_size[1] - 1, 1]]).repeat(enum, axis=0),
                             shape=(enum, 13), dtype=np.float32)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        num = self.agent_name_mapping[agent][0]
        # 将动作组合起来以方便处理
        # (技能1，技能2，技能q)，(站立，行走，奔跑），(移动，追击)  # 0表示不操作
        # 设置移动方位（地图大小）
        # 设置追击敌军
        return Box(low=np.zeros((num, 6)),
                    high=np.array([[3, 3, 2, self.world_size[0] - 1, self.world_size[1] - 1, self.agent_name_mapping[agent][1] - 1]]).repeat(num, axis=0),
                    shape=(num, 6), dtype=np.int32)

    def reset(self, seed=None, options=None):

        self.num_moves = 0
        self.agents = self.possible_agents[:]
        if self.render_mode == "human":
            import os
            # os.environ['SDL_VIDEODRIVER'] = 'windows'
            os.environ['SDL_VIDEODRIVER'] = 'x11'
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 200)
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Troy')
            pygame.display.set_icon(pygame.image.load("assert/icon.png"))
            gameMap = MainMap(self.map_path, self.window_size)
            gameGroup = MainGroup(gameMap.map_layer, self.window_size)
            self.clock = pygame.time.Clock()
        else:
            gameMap = SimpleMap(self.map_path)
            gameGroup = MainGroup()

        # 初始化环境
        t0 = troy_soldiers.Infrantry((420, 420), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t1 = troy_soldiers.Infrantry((420, 480), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t2 = troy_soldiers.Infrantry((420, 540), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t3 = troy_soldiers.Infrantry((420, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t4 = troy_soldiers.Infrantry((420, 660), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t5 = troy_soldiers.Infrantry((420, 720), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t6 = troy_soldiers.Infrantry((420, 780), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t7 = troy_soldiers.Infrantry((420, 840), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t8 = troy_soldiers.Rider((800, 40), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t9 = troy_soldiers.Rider((860, 40), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t10 =troy_soldiers.Rider((920, 40), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t11 = troy_soldiers.Rider((960, 40), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t12 = troy_soldiers.Archer((320, 420), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t13 = troy_soldiers.Archer((320, 480), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t14 = troy_soldiers.Archer((320, 540), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t15 = troy_soldiers.Archer((320, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t16 = troy_soldiers.Archer((320, 660), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t17 = troy_soldiers.Archer((320, 720), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t18 = troy_soldiers.Archer((320, 840), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t19 = troy_soldiers.TroyGuard((350, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t20 = troy_soldiers.TroyGuard((350, 840), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
        t21 = troy_soldiers.Hector((350, 450), gameGroup=gameGroup, gameMap=gameMap, which_troop=0,
                             is_hero=True)
        t22 = troy_soldiers.Priam((150, 450), gameGroup=gameGroup, gameMap=gameMap, which_troop=0,
                            is_king=True)

        g0 = greece_soldiers.GreeceArcher((900, 480), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g1 = greece_soldiers.GreeceArcher((900, 540), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g2 = greece_soldiers.GreeceArcher((900, 600), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g3 = greece_soldiers.GreeceArcher((900, 660), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
        g4 = greece_soldiers.AthensMan((950, 530), gameGroup=gameGroup, gameMap=gameMap,
                                  which_troop=1)
        g5 = greece_soldiers.AthensMan((950, 630), gameGroup=gameGroup, gameMap=gameMap,
                                  which_troop=1)
        g6 = greece_soldiers.MopoliaArcher((1040, 530), gameGroup=gameGroup, gameMap=gameMap,
                                      which_troop=1)
        g7 = greece_soldiers.MopoliaArcher((1040, 630), gameGroup=gameGroup, gameMap=gameMap,
                                      which_troop=1)
        g8 = greece_soldiers.MycenaeanSworder((820, 280), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g9 = greece_soldiers.MycenaeanSworder((820, 340), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g10 = greece_soldiers.MycenaeanSworder((820, 400), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g11 = greece_soldiers.MycenaeanSworder((820, 560), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g12 = greece_soldiers.MycenaeanSworder((820, 620), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g13 = greece_soldiers.MycenaeanSworder((640, 760), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g14 = greece_soldiers.MycenaeanSworder((640, 900), gameGroup=gameGroup, gameMap=gameMap,
                                         which_troop=1)
        g15 = greece_soldiers.ArgosMan((660, 320), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
        g16 = greece_soldiers.ArgosMan((660, 380), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
        g17 = greece_soldiers.PylosMan((920, 240), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
        g18 = greece_soldiers.PylosMan((980, 240), gameGroup=gameGroup, gameMap=gameMap,
                                 which_troop=1)
        g19 = greece_soldiers.SpartarMan((560, 770), gameGroup=gameGroup, gameMap=gameMap,
                                   which_troop=1)
        g20 = greece_soldiers.SpartarMan((560, 870), gameGroup=gameGroup, gameMap=gameMap,
                                   which_troop=1)
        g21 = greece_soldiers.Achilles((640, 820), gameGroup=gameGroup, gameMap=gameMap,
                                 which_troop=1, is_hero=True)
        g22 = greece_soldiers.AkaMenon((1000, 580), gameGroup=gameGroup, gameMap=gameMap,
                                 which_troop=1, is_king=True)

        self.gameMap = gameMap
        self.gameGroup = gameGroup
        self.troy_troop = [t0, t1, t2, t3, t4, t5, t6, t7,
                           t8, t9, t10, t11, t12, t13, t14, t15,
                           t16, t17, t18, t19, t20, t21, t22]
        self.greek_troop = [ g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14,
            g15, g16, g17, g18, g19, g20, g21, g22]

        observation = self._get_obs()
        info = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _get_obs(self):
        troy_ = []
        greek_sees = [] # 允许敌方获得一定的信息
        troy_num = self.agent_name_mapping[self.agents[0]][0]
        troy_mask = np.ones((troy_num, 6))
        for ti in range(troy_num):
            t = self.troy_troop[ti]
            # 移动状态，逃跑中，有效战斗，逃离战斗，不受限移动，被围攻，被夹击，处在实际战斗中
            # 士气，人数，体力，近战攻击，近战防御，冲锋加成，远程防御
            # 当前游戏时间
            if t.out_of_game:
                troy_.append(np.ones(20) * -1)
                greek_sees.append(np.ones(13) * -1)
            else:
                troy_.append(np.array([t.current_moving_state, t.current_flee_state,
                                   t.force_effect_fighting, t.moving_from_combat, t.no_strict_moving,
                                   t.in_besieged, t.in_pinched, t.in_actual_combat, t.current_hurt_by_remote,
                                    t.current_morale, t.current_soldier_nums, t.current_strength,
                                   t.current_close_combat_attack, t.current_close_combat_defense,
                                   t.current_charge_add, t.current_remote_combat_defense,
                                   *t.world_position,
                                       t.in_remote_combat() if isinstance(t, RemoteSoldier) else 0,
                                       t.current_ammo_num if isinstance(t, RemoteSoldier) else 0]))
                greek_sees.append(np.array([t.current_moving_state, t.current_flee_state, t.force_effect_fighting, t.moving_from_combat,
                                       t.no_strict_moving, t.in_besieged, t.in_pinched, t.in_actual_combat, t.current_hurt_by_remote,
                                       t.current_soldier_nums, *t.world_position, t.in_remote_combat() if isinstance(t, RemoteSoldier) else 0]))
            # 根据观测设置动作掩码，当离开游戏时，所有命令全部失效
            if t.out_of_game or t.current_flee_state >= 1:
                troy_mask[ti] = 0

        greek_ = []
        troy_sees = []
        greek_num = self.agent_name_mapping[self.agents[1]][0]
        greek_mask = np.ones((greek_num, 6))
        for ti in range(greek_num):
            t = self.greek_troop[ti]
            if t.out_of_game:
                greek_.append(np.ones(20) * -1)
                troy_sees.append(np.ones(13) * -1)
            else:
                greek_.append(np.array([t.current_moving_state, t.current_flee_state,
                                    t.force_effect_fighting, t.moving_from_combat, t.no_strict_moving,
                                    t.in_besieged, t.in_pinched, t.in_actual_combat, t.current_hurt_by_remote,
                                         t.current_morale, t.current_soldier_nums, t.current_strength,
                                       t.current_close_combat_attack, t.current_close_combat_defense,
                                       t.current_charge_add, t.current_remote_combat_defense,
                                    *t.world_position, t.in_remote_combat() if isinstance(t, RemoteSoldier) else 0,
                                        t.current_ammo_num if isinstance(t, RemoteSoldier) else 0]))
                troy_sees.append(np.array([t.current_moving_state, t.current_flee_state, t.force_effect_fighting, t.moving_from_combat,
                                       t.no_strict_moving, t.in_besieged, t.in_pinched, t.in_actual_combat, t.current_hurt_by_remote,
                                       t.current_soldier_nums, *t.world_position, t.in_remote_combat() if isinstance(t, RemoteSoldier) else 0]))
            if t.out_of_game or t.current_flee_state >= 1:
                greek_mask[ti] = 0

        return {
                self.agents[0]: {"mask": troy_mask,
                                 "obs_time": self.num_moves,
                                 "obs_self": np.array(troy_),
                                 "obs_enemy": np.array(troy_sees)},
                self.agents[1]: {"mask": greek_mask,
                                 "obs_time": self.num_moves,
                                 "obs_self": np.array(greek_),
                                 "obs_enemy": np.array(greek_sees)}
                }

    def render(self):
        # 显示模式
        if self.render_mode == "human":
            # 画面更新
            self.gameGroup.draw(self.window)
            pygame.display.flip()
            pygame.event.pump()
            # 允许交互
            if self.event_handler is None:
                event_handle(self.gameGroup, self.gameMap)
            else:
                self.event_handler(self.gameGroup, self.gameMap)
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            return None

    def step(
        self, action: ActType
    ) :
        ## 执行所有玩家的命令
        if action.get(self.agents[0]) is not None:
            troy_command = action[self.agents[0]][:, :3].astype(int)
            troy_move = action[self.agents[0]][:, 3:5].astype(int)
            troy_chase = action[self.agents[0]][:, 5].astype(int)
            for i in range(len(self.troy_troop)):
                s = self.troy_troop[i]
                if s.out_of_game:
                    continue
                abc = troy_command[i][0]
                if abc == 1:
                    s.command_ability1()
                elif abc == 2:
                    s.command_ability2()
                elif abc == 3:
                    s.command_abilityq()
                mc = troy_command[i][1]
                if mc > 0:
                    s.command_moving_state(mc - 1)
                mmc = troy_command[i][2]
                if mmc == 1:
                    s.command_chasing_enemy(self.greek_troop[troy_chase[i]])
                elif mmc == 2:
                    s.command_moving_position(troy_move[i])
        else:
            troy_command = np.array([0])

        if action.get(self.agents[1]) is not None:
            greek_command = action[self.agents[1]][:, :3].astype(int)
            greek_move = action[self.agents[1]][:, 3:5].astype(int)
            greek_chase = action[self.agents[1]][:, 5].astype(int)
            for j in range(len(self.greek_troop)):
                s = self.greek_troop[j]
                if s.out_of_game:
                    continue
                abc = greek_command[j][0]
                if abc == 1:
                    s.command_ability1()
                elif abc == 2:
                    s.command_ability2()
                elif abc == 3:
                    s.command_abilityq()
                mc = greek_command[j][1]
                if mc > 0:
                    s.command_moving_state(mc - 1)
                mmc = greek_command[j][2]
                if mmc == 1:
                    s.command_chasing_enemy(self.troy_troop[greek_chase[j]])
                elif mmc == 2:
                    s.command_moving_position(greek_move[j])
        else:
            greek_command = np.array([0])
        # 更新状态与跳出判断
        self.gameGroup.update()

        # 返回值记录
        observation = self._get_obs()
        info = {agent: {} for agent in self.agents}

        ##### 奖励设定
        rewards = {}
        # 敌人单位数量越多，自身奖励越少；自身单位数量越多，自身奖励越多
        # 尽量规避多余的行动
        # 尽量速战速决
        # 鼓励两组势力的靠近
        # troy_num = len(self.gameGroup.soldier_manager.troops[0]["troop"])
        # greek_num = len(self.gameGroup.soldier_manager.troops[1]["troop"])
        # troy_addition = (observation[self.agents[0]]["obs_self"][:, 9] * (observation[self.agents[0]]["obs_self"][:, 1] != 2)).sum() ** (1/2)+ \
        #                 (observation[self.agents[0]]["obs_self"][:, 10] * (observation[self.agents[0]]["obs_self"][:, 1] != 2)).sum() ** (1/2)
        # greek_addition = (observation[self.agents[1]]["obs_self"][:, 9] * (observation[self.agents[1]]["obs_self"][:, 1] != 2)).sum() ** (1/2)+ \
        #                  (observation[self.agents[1]]["obs_self"][:, 10] * (observation[self.agents[1]]["obs_self"][:, 1] != 2)).sum() ** (1/2)
        #
        troy_num = (observation[self.agents[0]]["obs_self"][:, 1] == 0).sum()
        greek_num = (observation[self.agents[1]]["obs_self"][:, 1] == 0).sum()
        g_c = [] # 记录未逃跑单位
        for g in self.gameGroup.soldier_manager.troops[0]["troop"]:
            if g.current_flee_state == 0:
                g_c.append(g.world_position)
        t_c = []
        for t in self.gameGroup.soldier_manager.troops[1]["troop"]:
            if t.current_flee_state == 0:
                t_c.append(t.world_position)
        dis = array_distance(np.array(g_c), np.array(t_c)).mean()    # control dis penalty
        #
        # troy_reward = -1e-4/(troy_num + troy_addition + (self.agent_name_mapping[self.agents[1]][0] - greek_num)**3 + 1e-7) - 1e-7 * (greek_addition + dis + np.sum(troy_command.astype(bool)) ** (1/2))
        # greek_reward = -1e-4/(greek_num + greek_addition + (self.agent_name_mapping[self.agents[0]][0]- troy_num)**3 + 1e-7) - 1e-7 * (troy_addition + dis + np.sum(greek_command.astype(bool)) ** (1/2))
        troy_reward = - 1e-7 * (dis + greek_num ** 2 + np.sum(troy_command.astype(bool)))
        greek_reward = - 1e-7 * (dis + troy_num ** 2 + np.sum(greek_command.astype(bool)))
        # troy_reward = - 1e-7 * (greek_num ** 2 + np.sum(troy_command.astype(bool)))
        # greek_reward = - 1e-7 * (troy_num ** 2 + np.sum(greek_command.astype(bool)))
        #
        did = self.gameGroup.soldier_manager.check_any_troop_destroyed()
        if did == 1:   # 击败敌人获得最大奖励，失败了不会获得惩罚，但是超时就都会获得最大惩罚
            terminations = {agent: True for agent in self.agents}
            rewards[self.agents[0]], rewards[self.agents[1]] = [1, -0.5]
            print("troy win!")
        elif did == 0:
            terminations = {agent: True for agent in self.agents}
            rewards[self.agents[0]], rewards[self.agents[1]] = [-0.5, 1]
            print("greek win!")
        else:
            rewards[self.agents[0]], rewards[self.agents[1]] = [troy_reward, greek_reward]
            terminations = {agent: False for agent in self.agents}

        # 可以设置游戏时间
        self.num_moves += 1
        env_truncation = (self.num_moves >= NUM_ITERS)
        if env_truncation:
            rewards[self.agents[0]] = rewards[self.agents[0]] - 1
            rewards[self.agents[1]] = rewards[self.agents[1]] - 1
            print("reach max game time limit!")
        truncations = {agent: env_truncation for agent in self.agents}

        # 渲染界面
        self.render()

        if self.num_moves % 100 == 0:   # 定时打印
            print(f"num_moves: {self.num_moves} distance: {dis} rewards: {rewards}")

        return observation, rewards, terminations, truncations, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed):
        np.random.seed(seed)

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
