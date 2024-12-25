import math
import os
import random
from itertools import zip_longest

import pygame
import numpy as np

from config_assert import ASSERT_DIR
from managers.morale_manager import MoraleManager
from utils.common import sigmoid, set_in_range, direction_check, sign2dvect, cal_distance, spiral_search, unsigned_to_signed_32_bit
from utils.pathfinder.graph.node import Node
from utils.pathfinder.solver.base import findPathBase
from utils.pathfinder.utils.path import makePath
from config_values import *

### 任何跟显示有关的都不要在这里处理！
### 外部类应尽量不与其属性直接交互

class Soldier(pygame.sprite.Sprite):

    name = "士兵"
    basic_walk_speed = 1    # px/每帧
    basic_run_speed = 2
    basic_close_combat_attack = 1
    basic_close_combat_defense = 1
    basic_charge_add = 0.01  # 冲锋加成

    basic_remote_combat_defense = 1

    basic_morale = 10
    basic_soldier_nums = 500
    basic_strength = 10 # 体力
    basic_stay_from_battle_time = 20    # 这么长时间没有经历战斗，则可以避免伤亡打击惩罚

    basic_abilities = []

    def __init__(self, world_position, gameGroup, gameMap, scale_size=(50, 50), portrait_path=None, main_color=None,
                 which_troop=0, is_king=False, is_hero=False):
        super(Soldier, self).__init__()

        # image
        self.image = self.preprocess_potrait(portrait_path, scale_size, main_color)
        self.rect = self.image.get_rect()
        self.main_color = main_color

        # 初始化位置
        self.show_human = False
        self._set_position(world_position)

        # 当前状态
        self.current_morale = self.basic_morale
        self.current_soldier_nums = self.basic_soldier_nums
        self.current_strength = self.basic_strength

        self.current_close_combat_attack = self.basic_close_combat_attack
        self.current_close_combat_defense = self.basic_close_combat_defense
        self.current_charge_add = self.basic_charge_add  # 冲锋加成
        self.current_remote_combat_defense = self.basic_remote_combat_defense

        self.out_of_game = 0    # 0表示在游戏中，1表示已经覆灭
        self.current_moving_state = 0  # 0表示站立，1表示行走，2表示奔跑，3表示冲锋
        self.current_flee_state = 0  # 0表示正常，1表示崩溃，2表示毁灭
        self.current_hurt_by_remote = False  # 每回合开始重置，有受到伤害则置此项
        self.current_under_command = 0  # 0表示未受指挥，1表示一般指挥，2表示高级指挥
        # 设置被动技能
        self.current_abilities = []
        for abl in self.basic_abilities:
            if abl[-1] == 0:
                self.current_abilities.append(abl)

        #### 非属性现状
        # 特殊能力
        self.force_effect_fighting = False  # 有些兵种能限制与其交战的单位最多有效士兵数量

        # 移动相关
        self.moving_position = []  # 移动路径（世界坐标，tiled索引）
        self.moving_direction = (0, 0)  # 移动方向，关系到冲锋（二元符号方向向量）
        self.chasing_target = None
        self.moving_from_combat = False
        self.no_strict_moving = False   # 某种情况下为不受限的移动，移动会破坏掉所有正处于的combat

        # 战斗现状
        self.joining_combates = []
        self.fighting_enemies = []
        self.in_besieged = False # 被围攻
        self.in_pinched = False # 被夹击
        self.in_actual_combat = False   # 当交战敌人全为已崩溃的敌人时，认为不在实际的战斗中

        # 士气
        self.morale_set = MoraleManager()

        # 时钟控制
        self.show_clock = 0
        self.combat_clock = 0   # 每次脱离战斗都应当重置此时钟
        self.charge_clock = 0   # 冲锋加成时间
        self.last_get_into_battle_clock = -self.basic_stay_from_battle_time  # 距离上次脱离战斗的时间，用于士气的处理

        # 缓存
        self.strength_propose = 0
        self.morale_propose = 0
        self.soldier_number_propose = 0
        self.distance_dict = {
            "arround_friendly": [],
            "arround_courage": [],
            "arround_threaten": [],
            "arround_enemy": [],
            "arround_command": False,
            "arround_command_order": False
        } # 本轮距离中角色记录字典

        #
        gameGroup.add_soldier(self, which_troop, is_king, is_hero)
        self.group = gameGroup
        self.gameMap = gameMap
        self.troop_id = which_troop

    def preprocess_potrait(self, portrait_path, scale_size, main_color, main_threshold=(30, 30, 30)):
        image = pygame.image.load(os.path.join(ASSERT_DIR, portrait_path))
        thold = image.map_rgb(main_threshold)  # 强制转换为无符号数
        if scale_size != None:
            image = pygame.transform.scale(image, scale_size)
        if main_color != None:
            with pygame.PixelArray(image) as pxarray:
                for i in range(image.get_width()):
                    for j in range(image.get_height()):
                        if unsigned_to_signed_32_bit(pxarray[i, j]) <= thold:    # 小于此值认为是主要区域（这里将所有值强制转换为有符号32位整数，避免操作系统不同出现的问题）
                            pxarray[i, j] = main_color
        return image

    def pppreprocess_turn(self):
        # 重置字典
        for k in self.distance_dict.keys():
            if type(self.distance_dict[k]) is bool:
                self.distance_dict[k] = False
            else:
                self.distance_dict[k] = []

    def ppreprocess_turn(self):
        ## 性能优化，所有涉及到遍历军队的操作在这里直接执行！
        # 周遭环境状态检查
        self_wp = self.world_position
        for e in self.group.soldier_manager.get_his_enemy_troop(self.troop_id):
            dis = cal_distance(e.world_position, self_wp)
            if dis <= ARROUND_INFLUENCE_DISTANCE:
                self.distance_dict["arround_enemy"].append(e)
            if self.ability_enabled(THREATEN_ENEMY_ABILITY) and dis <= EFFECTIVE_THREATEN_RANGE:
                e.distance_dict["arround_threaten"].append((self, dis))
        for s in self.group.soldier_manager.get_his_troop(self.troop_id):
            dis = cal_distance(s.world_position, self_wp)
            if self.ability_enabled(COURAGE_ABILITY) and dis <= EFFECTIVE_COURAGE_RANGE:
                s.distance_dict["arround_courage"].append((self, dis))
            if s == self:
                continue
            if dis <= ARROUND_INFLUENCE_DISTANCE:
                self.distance_dict["arround_friendly"].append(s)
            if self.ability_enabled(COMMAND_ABILITY) and dis <= EFFECTIVE_COMMAND_RANGE:
                s.distance_dict["arround_command"] = True
                if self.ability_enabled(COMMAND_ORDER):
                    s.distance_dict["arround_command_order"] = True

    def preprocess_turn(self):  # 每回合开始前最早的准备工作，设置一系列状态位
        self.current_hurt_by_remote = False
        self._check_under_command()
        # 追击敌军移动序列设置
        # 追逐目标不合法
        if self.chasing_target != None:
            if not self.chasing_target in self.group.soldiers:
                self.command_moving_state(0)
            else:
                self._get_target_position_path_sequence(self.chasing_target.rect.center)  # 立刻生成移动序列
        # 回合开始时设置当前移动方向
        if len(self.moving_position) > 0:
            # 设置方向
            self.moving_direction = sign2dvect((self.moving_position[0][0] - self.world_position[0], self.moving_position[0][1] - self.world_position[1]))
        else:
            self.moving_direction = (0, 0)
            self.moving_from_combat = False # 没有移动目的地，终止退出战斗
        # 根据移动方向判断是否被阻塞
        if self.in_close_combat():
            if not self.is_surrending() and self._check_stuck_in_battle():
                self.moving_from_combat = False
            # 保持计时器
            self.last_get_into_battle_clock = self.group.clock
            # 检查战斗状态，这里的状态会影响到后续士气、体力、人数等的计算（被围攻、真实战斗...）
            self._check_current_fight()
        # 尝试脱离战斗（一定要在位置变化前进行，否则会一直无法脱离战斗）
        if self.moving_from_combat:
            for combat in self.joining_combates.copy():
                combat.chaos -= 2
                if combat.chaos <= 0:  # 脱离当前战斗
                    self.group.remove_combat(combat)
        # 可移动与否的状态位
        self._check_no_strict_moving()

    def propose(self):  # 只进行提议，不最终更新值（确保公平），在所有单位提议完毕后统一更新。并且设置一些在进过group的”战斗更新“后才能确定的状态。
        # 战斗检测
        if self.in_close_combat():
            # 更新人数，这里我们采用一个累计时钟机制，在战斗中时，时钟变量会不断增大，当到达某个值时才会进行人数更新并重置时钟。即使中途退出了战斗也会保持时钟。
            if self.is_combat_update():
                self._propose_soldier_nums_v2()
        else:
            if self.current_hurt_by_remote:
                self.last_get_into_battle_clock = self.group.clock
            self.moving_from_combat = False  # 未在战斗中，不应当逃离
            self.in_actual_combat = False   # 未在受限战斗中
        # 更新体力（仅仅更新了值，没有考虑体力等级问题）
        self._propose_strength()
        # 更新士气，如果被摧毁了就再也不用更新士气了
        if not self.is_destroyed():
            self._propose_morale()

    def update(self) -> None:    # 必须在循环的最后一步。在士气、人数更新后进行后续检查。
        # 逃离战场
        if self.is_surrending():
            # 是否逃离战场，在一开始进行检测，避免后续bug
            if (self.world_position[0] < 25 or self.world_position[0] >
                                          self.gameMap.world_size[0] - 26 or self.world_position[1] < 25 or
                                          self.world_position[1] > self.gameMap.world_size[1] - 26):
                self.set_out_of_game()
                return
            if self.in_close_combat():
                self.moving_from_combat = True
            if len(self.moving_position) == 0:
                self._flee_from_map()
        # 更新位置
        if len(self.moving_position) > 0 and self.no_strict_moving:
            # 尝试删除所有关联战斗 （这是必要的，当不限制移动时）
            for combat in self.joining_combates.copy():
                self.group.remove_combat(combat)
            # 位置更新
            self._update_position()
            # 战斗检查
            for espr in self.distance_dict["arround_enemy"]:
                if espr.out_of_game == 0 and espr.rect.colliderect(self.rect):  # 进入交战状态。加上out_of_game的检测，因为可能在本轮阵亡
                    # 位置
                    collision_rect = espr.rect.clip(self.rect)
                    collision_x = collision_rect.x
                    collision_y = collision_rect.y
                    # 创建combat
                    if not (self.is_surrending() and espr.is_surrending()):
                        self.group.add_combat(self, espr, (collision_x, collision_y))

    def _check_current_fight(self): # 检查当前战斗局面
        # 受敌情况
        vects_x = set()
        vects_y = set()
        vects_all = set()
        for enemy in self.fighting_enemies:
            if enemy.is_surrending():
                continue
            vect = (
                enemy.world_position[0] - self.world_position[0], enemy.world_position[1] - self.world_position[1])
            vects_x.add(vect[0] / abs(vect[0]) if vect[0] != 0 else 0)
            vects_y.add(vect[1] / abs(vect[1]) if vect[1] != 0 else 0)
            if vect[0] != 0 and vect[1] != 0:
                vects_all.add((vect[0] / abs(vect[0]), vect[1] / abs(vect[1])))
            # 敌人技能影响
            if self.ability_enabled(IN_DENSE) and (enemy.ability_enabled(STRIKE_ATTACK) or enemy.ability_enabled(BROKE_DENSE_ABILITY)):
                self.command_ability1()
        if len(vects_all) >= 4:
            self.in_besieged = True
            # 破坏密集阵型
            if self.ability_enabled(IN_DENSE):
                self.current_abilities.remove(IN_DENSE)
        else:
            self.in_besieged = False
        # 夹击判断
        if ((1 in vects_x and -1 in vects_x and (len(vects_y) > 1 or vects_y == [0])) or
                (1 in vects_y and -1 in vects_y and (len(vects_x) > 1) or vects_x == [0])):
            self.in_pinched = True
            # 破坏密集阵型
            if self.ability_enabled(IN_DENSE):
                self.current_abilities.remove(IN_DENSE)
        else:
            self.in_pinched = False
        # 是否与逃兵交战
        if len(vects_x) != 0 or len(vects_y) != 0:
            self.in_actual_combat = True
        else:
            self.in_actual_combat = False
        # 特殊能力
        if self.ability_enabled(IN_DENSE) or self.ability_enabled(IN_TESTUDO):
            self.force_effect_fighting = True
        else:
            self.force_effect_fighting = False
    def _check_no_strict_moving(self):
        if (self.in_actual_combat and not self.ability_enabled(STRIKE_ATTACK)) or self.ability_enabled(REMOTE_ATTACK_ABILITY):
            self.no_strict_moving = False
        else:
            self.no_strict_moving = True
    def _check_under_command(self):
        if self.distance_dict["arround_command_order"]:
            self.current_under_command = 2
        elif self.distance_dict["arround_command"]:
            self.current_under_command = 1
        else:
            self.current_under_command = 0
            return 1
        # 处于指挥状态中
        ratio = COMMAND_ORDER_MULTIPLT_RATIO if self.current_under_command == 2 else COMMAND_MULTIPLY_RATIO
        self.current_close_combat_attack = self.basic_close_combat_attack * ratio
        self.current_close_combat_defense = self.basic_close_combat_defense * ratio
        self.current_charge_add = self.basic_charge_add * ratio # 冲锋加成
        self.current_remote_combat_defense = self.basic_remote_combat_defense * ratio
        return ratio

    def _check_stuck_in_battle(self):
        v = np.array(self.moving_direction)
        nvs = []
        for cmb in self.joining_combates:
            nvs.append(np.array(sign2dvect((cmb.world_position[0] - self.world_position[0], cmb.world_position[1] - self.world_position[1]))))
        nvs = np.array(nvs)
        if direction_check(v, nvs):  # 若至少有一个同向，则陷入战斗，不可逃离
            return True
        else:
            return False

    def propose_soldier_nums_encounter(self, enemy, encounter_position):    # 首次交战

        # 交战位置与冲锋方向
        this_combat_direction = (encounter_position[0] - self.world_position[0], encounter_position[1] - self.world_position[1])
        enemy_combat_direction = (encounter_position[0] - enemy.world_position[0], encounter_position[1] - enemy.world_position[1])
        # 归一化，取符号
        this_combat_direction = sign2dvect(this_combat_direction)
        enemy_combat_direction = sign2dvect(enemy_combat_direction)
        if this_combat_direction[0] * self.moving_direction[0] + this_combat_direction[1] * self.moving_direction[1] > 0:
            charge_this = self.charge_clock * self.current_charge_add    # 有效
        else:
            charge_this = 0
        if enemy_combat_direction[0] * enemy.moving_direction[0] + enemy_combat_direction[1] * enemy.moving_direction[1] > 0:
            charge_enemy = enemy.charge_clock * enemy.current_charge_add
        else:
            charge_enemy = 0

        # 己方战力
        eman_this = self.get_effective_man() # 战斗的发起方是
        mratio_this = sigmoid(self.current_morale / MORALE_DROP_STANDARD)  # 士气
        sratio_this = sigmoid(self.current_strength / STRENGTH_DROP_STANDARD)  # 体力
        penalty_this = PENALTY_MOVING_FROM_BATTLE if self.moving_from_combat else 1  # 脱离战斗惩罚
        if self.is_destroyed():
            surrending_this = 0.5
        elif self.is_surrending():
            surrending_this = 0.85
        else:
            surrending_this = 1
        random_this = random.random() * 0.01 + 0.99  # 随机惩罚
        fabl_this = set_in_range(math.log(eman_this), lower_bound=1) * (self.current_attack_close() + charge_this) * mratio_this * penalty_this * sratio_this * surrending_this * random_this

        eman_enem_suggest = enemy.get_effective_man()
        if self.force_effect_fighting:
            eman_enem = eman_this if eman_enem_suggest > eman_this else eman_enem_suggest
        else:
            eman_enem = eman_enem_suggest
        mratio_enem = sigmoid(enemy.current_morale / MORALE_DROP_STANDARD)
        sratio_enem = sigmoid(enemy.current_strength / STRENGTH_DROP_STANDARD)
        penalty_enem = PENALTY_MOVING_FROM_BATTLE if enemy.moving_from_combat else 1
        if enemy.is_destroyed():
            surrending_enem = 1e-5
        elif enemy.is_surrending():
            surrending_enem = 2e-3
        else:
            surrending_enem = 1
        random_enem = random.random() * 0.01 + 0.99
        fabl_enem = set_in_range(math.log(eman_enem), lower_bound=1) * (enemy.current_attack_close() + charge_enemy) * mratio_enem * penalty_enem * sratio_enem * surrending_enem * random_enem

        # 实际减损
        # hurt = (fabl_enem - fabl_this / fabl_enem * eman_this) / (self.current_defense_close() * eman_this)
        hurt = (fabl_enem ** 2) / (self.current_defense_close() * math.log(eman_this) * fabl_this) * \
               sigmoid(math.log10(fabl_enem / fabl_this)) * set_in_range(math.log(eman_this, 100), lower_bound=1)
        hurt = -hurt if hurt > 0 else -random.random() * 0.001
        self.soldier_number_propose += hurt

    def propose_hurt_by_arrows(self, archer, area):
        e_abl = archer.current_remote_attack() * set_in_range(math.log(archer.current_soldier_nums), lower_bound=1)
        s_abl = self.current_remote_defense()
        hurt = (e_abl/s_abl) * set_in_range(math.log(self.current_soldier_nums), lower_bound=1) * area / (self.rect[-1] * self.rect[-2])  # 人数越多伤的越多，考虑重叠面积
        self.soldier_number_propose += -hurt if hurt > 0 else -random.random()*0.001

    def _propose_soldier_nums(self):
        eman_this = self.get_effective_man()
        mratio_this = sigmoid(self.current_morale / MORALE_DROP_STANDARD)   # 士气
        sratio_this = sigmoid(self.current_strength / STRENGTH_DROP_STANDARD)    # 体力
        penalty_this = PENALTY_MOVING_FROM_BATTLE if self.moving_from_combat else 1  # 脱离战斗惩罚
        if self.is_destroyed():
            surrending_this = 1e-7
        elif self.is_surrending():
            surrending_this = 0.0002
        else:
            surrending_this = 1
        random_this = random.random() * 0.01 + 0.99  # 随机惩罚
        fabl_this = eman_this * self.current_attack_close() * mratio_this * penalty_this * sratio_this * surrending_this * random_this

        fabl_enems = 0
        for e in self.fighting_enemies:
            eman_enem_suggest = e.get_effective_man()
            if self.force_effect_fighting:
                eman_enem = eman_this if eman_enem_suggest > eman_this else eman_enem_suggest
            else:
                eman_enem = eman_enem_suggest
            mratio_enem = sigmoid(e.current_morale / MORALE_DROP_STANDARD)
            sratio_enem = sigmoid(e.current_strength / STRENGTH_DROP_STANDARD)
            penalty_enem = PENALTY_MOVING_FROM_BATTLE if e.moving_from_combat else 1
            if e.is_destroyed():
                surrending_enem = 1e-7
            elif e.is_surrending():
                surrending_enem = 0.0002
            else:
                surrending_enem = 1
            random_enem = random.random() * 0.01 + 0.99
            fabl_enem = eman_enem * e.current_attack_close() * mratio_enem * penalty_enem * sratio_enem * surrending_enem * random_enem
            # 加和
            fabl_enems += fabl_enem

        # 实际减损
        hurt = (fabl_enems - fabl_this/fabl_enems*eman_this)  / (self.current_defense_close() * eman_this)
        hurt = -hurt if hurt > 0 else -random.random()*0.001
        self.soldier_number_propose += hurt

    def _propose_soldier_nums_v2(self): # 改进逻辑
        eman_this = self.get_effective_man()
        mratio_this = sigmoid(self.current_morale / MORALE_DROP_STANDARD)  # 士气
        sratio_this = sigmoid(self.current_strength / STRENGTH_DROP_STANDARD)  # 体力
        penalty_this = PENALTY_MOVING_FROM_BATTLE if self.moving_from_combat else 1  # 脱离战斗惩罚
        if self.is_destroyed():
            surrending_this = 0.5
        elif self.is_surrending():
            surrending_this = 0.85
        else:
            surrending_this = 1
        random_this = random.random() * 0.01 + 0.99  # 随机惩罚
        fabl_this = set_in_range(math.log(eman_this), lower_bound=1) * self.current_attack_close() * mratio_this * penalty_this * sratio_this * surrending_this * random_this

        fabl_enems = 0
        for e in self.fighting_enemies:
            eman_enem_suggest = e.get_effective_man()
            if self.force_effect_fighting:
                # 如果是密集状态，为每个敌人分配一个平均战斗人数
                eman_this_average = eman_this / len(self.joining_combates) if len(self.joining_combates) > 0 else eman_this
                eman_enem = eman_this_average if eman_enem_suggest > eman_this_average else eman_enem_suggest
            else:
                eman_enem = eman_enem_suggest
            mratio_enem = sigmoid(e.current_morale / MORALE_DROP_STANDARD)
            sratio_enem = sigmoid(e.current_strength / STRENGTH_DROP_STANDARD)
            penalty_enem = PENALTY_MOVING_FROM_BATTLE if e.moving_from_combat else 1
            if e.is_destroyed():
                surrending_enem = 1e-5
            elif e.is_surrending():
                surrending_enem = 2e-3
            else:
                surrending_enem = 1
            random_enem = random.random() * 0.01 + 0.99
            fabl_enem = set_in_range(math.log(eman_enem), lower_bound=1) * e.current_attack_close() * mratio_enem * penalty_enem * sratio_enem * surrending_enem * random_enem
            # 加和
            fabl_enems += fabl_enem

        # 实际减损
        hurt = (fabl_enems ** 2) / (self.current_defense_close() * set_in_range(math.log(eman_this), lower_bound=1) * fabl_this) * \
               sigmoid(math.log10(fabl_enems / fabl_this)) * set_in_range(math.log(eman_this, 100), lower_bound=1) # 第二项调和控制，防止较大的战力差距，第三项为受伤概率
        hurt = -hurt if hurt > 0 else -random.random() * 0.001
        self.soldier_number_propose += hurt

    def _propose_strength(self):
        if self.in_close_combat():
            self.strength_propose += STRENGTH_COMBATING_LOSE
        else:
            if self.current_moving_state == 0:
                self.strength_propose += STRENGTH_STANDING_LOSE
            elif len(self.moving_position) > 0 and self.no_strict_moving:   # 仅在真实移动时才会计算
                if self.current_moving_state == 1:
                    self.strength_propose += STRENGTH_WALKING_LOSE
                elif self.current_moving_state == 2:
                    self.strength_propose += STRENGTH_RUNNING_LOSE
                elif self.current_moving_state == 3:
                    self.strength_propose += STRENGTH_CHARGING_LOSE

    def _propose_morale(self):
        self.morale_set.check_charge(self.current_moving_state)
        self.morale_set.check_continous_hurt(self.current_soldier_nums/self.basic_soldier_nums,
                                             self.last_get_into_battle_clock,
                                             self.group.clock,
                                             self.basic_stay_from_battle_time)
        self.morale_set.check_fight(self.in_besieged, self.in_pinched)
        self.morale_set.check_arround(self)
        self.morale_set.check_moving_from_combat(self.moving_from_combat, self.in_actual_combat)
        self.morale_set.check_strength(self.current_strength, STRENGTH_DROP_LEVEL4, STRENGTH_DROP_LEVEL3)
        self.morale_set.check_king_died(self.group.soldier_manager.get_his_king(self.troop_id), self.group.soldier_manager.get_his_kh_ratio(self.troop_id, 0))
        self.morale_set.check_courage(self.distance_dict["arround_courage"])
        if not self.ability_enabled(IMMUNE_THREAT):
            self.morale_set.check_threatened(self.distance_dict["arround_threaten"])
        self.morale_set.check_chasing_flee(self.in_actual_combat, self.joining_combates)
        self.morale_set.check_hero_dead(self.group.soldier_manager.get_his_hero(self.troop_id), self.group.soldier_manager.get_his_kh_ratio(self.troop_id, 1))
        # 更新士气
        for mr in self.morale_set.get_values():
            self.morale_propose += mr

    def _update_position(self):
        # 检查是否触发冲锋
        if self.chasing_target and len(self.moving_position) < CHARGE_PATH_NUMS:
            self.command_moving_state(3)
            self.charge_clock = set_in_range(self.charge_clock + 1, 0, CHARGE_PATH_NUMS)
        else:
            self.charge_clock = 0
        # 检查移动状态
        if self.current_moving_state > 1:
            speed = self.basic_run_speed * (self.current_strength / self.basic_strength)
        else:
            speed = self.basic_walk_speed * (self.current_strength / self.basic_strength)
        while speed > 1 and len(self.moving_position) > 1:
            self.moving_position.pop(0)
            speed -= 1
        moving_position = self.moving_position.pop(0)
        # 设置位置
        self._set_position(moving_position)

    def _set_position(self, world_position): # 屏幕移动也会导致屏幕坐标变化，但不会导致世界坐标变化，因此我们这里只设置跟世界坐标有关的部分
        self.world_position = world_position
        self.rect.center = self.world_position

    def _get_target_position_path_sequence(self, world_position):
        # 计算移动序列（逐像素）
        pathfinder = self.gameMap.solver
        access_map = self.gameMap.access_map
        tile_size = self.gameMap.tile_size

        spos = self.world_position
        dpos = world_position

        spos_tiled = spos[0] // tile_size[0], spos[1] // tile_size[1]
        # 若起点不幸被回归到墙壁，则尝试跳出（记得处理边界）
        t= 0
        while not (spos_tiled[0] < access_map.width and spos_tiled[1] < access_map.height) or access_map.cells[spos_tiled[0]][spos_tiled[1]]:
            t += 1
            # 螺旋探测，尝试跳出
            px, py = spiral_search(t)[:2]
            spos_tiled = (int(spos_tiled[0] + px), int(spos_tiled[1] + py))

        dpos_tiled = int(dpos[0] / tile_size[0]), int(dpos[1] / tile_size[1])

        # path = find_path(spos, dpos, access_map)
        snode = Node(*spos_tiled)
        dnode = Node(*dpos_tiled)
        found, last_node, closed_nodes, open_nodes = findPathBase(pathfinder, access_map, snode, dnode)
        if found:
            jump_path = makePath(last_node)
            #
            path = []
            # 重整成像素级移动（根据速度）
            last_ps = None
            for pi in range(len(jump_path)):
                if last_ps == None: # 起点
                    last_ps = spos
                    continue
                if pi == len(jump_path) - 1:    # 终点
                    this_ps = dpos
                else:
                    # 中间点
                    this_ps_tiled = jump_path[pi].tuple
                    # 转像素级
                    this_ps = this_ps_tiled[0] * tile_size[0], this_ps_tiled[1] * tile_size[1]
                # 根据不同行径方向进行调整
                if last_ps[0] > this_ps[0]:
                    xlist = list(range(last_ps[0], this_ps[0]-1, -1))
                else:
                    xlist = list(range(last_ps[0], this_ps[0]+1, 1))
                if last_ps[1] > this_ps[1]:
                    ylist = list(range(last_ps[1], this_ps[1]-1, -1))
                else:
                    ylist = list(range(last_ps[1], this_ps[1]+1, 1))
                path.extend(list(zip_longest(xlist, ylist,
                            fillvalue=this_ps[0] if abs(this_ps[0]-last_ps[0]) < abs(this_ps[1]-last_ps[1]) else this_ps[1])))  # 先走斜线再走直线
                last_ps = this_ps
            self.moving_position = path[1:] # 第一个值不当保存
        else:
            self.moving_position = []

    def _flee_from_map(self):
        # 尝试随机出口
        x_max = self.gameMap.world_size[0] - 1
        y_max = self.gameMap.world_size[1] - 1
        if random.random() > 0.5:
            x = random.choice([0, x_max])
            y = random.randint(0, y_max)
        else:
            x = random.randint(0, x_max)
            y = random.choice([0, y_max])
        self._get_target_position_path_sequence((x, y))

    ## 状态判断
    def current_attack_close(self):
        if self.ability_enabled(IN_DENSE):
            return self.current_close_combat_attack * IN_DENSE_ATTACK_MULTI_RATIO
        elif self.ability_enabled(IN_TESTUDO):
            return self.current_close_combat_attack * IN_TESTUDO_ATTACK_MULTI_RATIO
        elif self.ability_enabled(STRIKE_ATTACK):
            return self.current_close_combat_attack * STRIKE_ATTACK_MULTI_RATIO
        else:
            return self.current_close_combat_attack
    def current_defense_close(self):
        if self.ability_enabled(IN_DENSE):
            return self.current_close_combat_defense * IN_DENSE_DEFENSE_MULTI_RATIO
        elif self.ability_enabled(IN_TESTUDO):
            return self.current_close_combat_defense * IN_TESTUDO_DEFENSE_MULTI_RATIO
        elif self.ability_enabled(STRIKE_ATTACK):
            return self.current_close_combat_defense * STRIKE_DEFENSE_MULTI_RATIO
        else:
            return self.current_close_combat_defense
    def current_remote_defense(self):
        # 如果是龟甲阵，增强远程防御
        ratio = 1
        if self.ability_enabled(IN_TESTUDO):
            ratio += 0.2
        if self.ability_enabled(IN_DENSE):
            ratio -= 0.2
        # 如果在近战中，将会降低对远程攻击的防御能力
        if self.in_close_combat():
            return self.current_remote_combat_defense / len(self.joining_combates) * ratio
        else:
            return self.current_remote_combat_defense * ratio

    def get_effective_man(self):
        if self.ability_enabled(IN_DENSE):
            return EFFECTIVE_FIGHTER_IN_DENSE if self.current_soldier_nums > EFFECTIVE_FIGHTER_IN_DENSE else self.current_soldier_nums
        elif self.ability_enabled(IN_TESTUDO):
            return EFFECTIVE_FIGHTER_IN_TESTUDO if self.current_soldier_nums > EFFECTIVE_FIGHTER_IN_TESTUDO else self.current_soldier_nums
        else:
            return self.current_soldier_nums / len(self.joining_combates) if len(self.joining_combates) > 0 else self.current_soldier_nums
    def get_strength_level(self):
        if self.current_strength >= STRENGTH_DROP_LEVEL1:
            return 0
        elif self.current_strength >= STRENGTH_DROP_LEVEL2:
            return 1
        elif self.current_strength >= STRENGTH_DROP_LEVEL3:
            return 2
        elif self.current_strength >= STRENGTH_DROP_LEVEL4:
            return 3
        else:
            return 4

    def is_combat_update(self):
        self.combat_clock += 1
        if self.combat_clock >= COMBAT_RESULTING_TIME:
            self.combat_clock = 0
            return True
        else:
            return False
    def is_show_update(self):
        self.show_clock += 1
        if self.current_flee_state == 1:
            clock = SHOW_CLOCK_SURRENDING
        else:
            clock = SHOW_CLOCK_NORMAL
        if self.show_clock >= clock:
            self.show_clock = 0
            return True
        else:
            return False

    def in_close_combat(self):
        return len(self.joining_combates) > 0
    def set_in_close_combat(self, enemy, combat):   # 仅在生成战斗时触发
        self.charge_clock = 0
        self.joining_combates.append(combat)
        self.fighting_enemies.append(enemy)
        if self.is_surrending():
            self.moving_from_combat = True
        if not self.moving_from_combat: # 终止移动
            self.command_moving_state(0)    # 站立并停止追击

    def is_surrending(self):
        return self.current_flee_state != 0
    def set_surrending(self):
        self.current_flee_state = 1 # 置为崩溃
        self.command_moving_state(2)  # 逃跑状态
        # 取消原本的行动
        self.moving_position = []
        self.chasing_target = None
        # 单位崩溃后，失去所有特殊能力
        self.current_abilities = []
    def is_destroyed(self):
        return self.current_flee_state == 2
    def set_destroyed(self):
        self.current_flee_state = 2
        # 清空士气影响
        self.morale_set.clear()
    def set_recover(self):
        self.current_flee_state = 0  # 设为正常
        self.command_moving_state(0)
        # 重拾所有能力
        self.current_abilities = list(self.basic_abilities)
    def set_out_of_game(self):  # 逃离战场、或被彻底消灭，设此项
        self.out_of_game = 1
        self.group.remove_soldier(self)

    def ability_enabled(self, ability_name):
        return ability_name in self.current_abilities

    # flush操作应为循环的结束操作
    def flush_all(self):
        # 更新所有缓存变量
        ###### 人数
        self.current_soldier_nums += self.soldier_number_propose
        if self.current_soldier_nums <= 0:  # 已被消灭
            self.set_out_of_game()
            return
        ###### 体力
        self.current_strength += self.strength_propose
        self.current_strength = set_in_range(self.current_strength, 0, self.basic_strength)
        ###### 士气
        self.current_morale += self.morale_propose
        self.current_morale = set_in_range(self.current_morale, 0, self.basic_morale)
        # 崩溃检查
        if self.current_morale < MORALE_DROP_LEVEL3 and not self.is_surrending():
            self.set_surrending()
        # 被摧毁检查
        if self.current_morale < MORALE_DROP_LEVEL4:
            self.set_destroyed()
        # 回归检查
        if self.is_surrending() and self.current_morale >= MORALE_DROP_LEVEL3:
            self.set_recover()
        ###### 重置状态
        self.strength_propose = 0
        self.morale_propose = 0
        self.soldier_number_propose = 0
    def flush_number(self):
        self.current_soldier_nums += self.soldier_number_propose
        self.soldier_number_propose = 0
        if self.current_soldier_nums <= 0:  # 已被消灭
            self.set_out_of_game()

    ################# 可执行的命令，应仅涉及简单的状态设置
    def command_chasing_enemy(self, e):
        if self.is_surrending():
            return
        self.chasing_target = e
        self.command_moving_state(1)

        # 如果追踪的对象为正在交战的敌人，则取消脱离战斗
        if self.in_close_combat():
            if e in self.fighting_enemies:
                self.moving_from_combat = False
            else:
                self.moving_from_combat = True

    def command_moving_position(self, world_position):
        if self.is_surrending():
            return
        if self.in_close_combat():
            self.moving_from_combat = True
        self.chasing_target = None
        self._get_target_position_path_sequence(world_position)
        self.command_moving_state(1)

    def command_moving_state(self, state_code):
        if self.is_surrending() or self.ability_enabled(REMOTE_ATTACK_ABILITY): # 当处于射击状态时，取消移动状态改变
            return
        if state_code == 0:  # 取消之前的移动序列
            if not self.ability_enabled(STRIKE_ATTACK):
                self.moving_position = []
                self.chasing_target = None
                if hasattr(self, "chasing_target_remote_attack"):
                    self.chasing_target_remote_attack = None
        if state_code > 1:  # 跑步破坏掉密集阵型和龟甲阵型
            if self.ability_enabled(IN_DENSE) or self.ability_enabled(IN_TESTUDO):
                self.command_ability1()
        self.current_moving_state = state_code

    def command_ability1(self):
        if self.is_surrending():
            return
        self._command_ability1()

    def command_ability2(self):
        if self.is_surrending():
            return
        self._command_ability2()

    def command_abilityq(self):
        if self.is_surrending():
            return
        self._command_abilityq()

    ####
    def _command_ability1(self):
        pass

    def _command_ability2(self):
        pass

    def _command_abilityq(self):
        pass