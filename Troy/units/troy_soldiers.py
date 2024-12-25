import os

from units.remote_soldier import RemoteSoldier
from units.soldier import Soldier
from config_values import *

class Infrantry(Soldier):

    name = "特洛伊步兵"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 2
    basic_close_combat_attack = 45
    basic_close_combat_defense = 50
    basic_charge_add = 0.02  # 冲锋加成

    basic_remote_combat_defense = 70

    basic_morale = 10
    basic_soldier_nums = 500
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 20  # 脱离战斗后恢复时间

    basic_abilities = (IN_DENSE, )
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a13.png"), main_color=(0, 200, 255), **kwargs):
        super(Infrantry, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

    # 密集阵型
    def _command_ability1(self):
        if self.ability_enabled(IN_DENSE):
            self.current_abilities.remove(IN_DENSE)
        else:
            self.current_abilities.append(IN_DENSE)

class Rider(Soldier):
    
    name = "特洛伊骑手"

    basic_walk_speed = 3 # px/每帧
    basic_run_speed = 5
    basic_close_combat_attack = 60
    basic_close_combat_defense = 45
    basic_charge_add = 0.8  # 冲锋加成

    basic_remote_combat_defense = 30

    basic_morale = 10
    basic_soldier_nums = 200
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 20

    basic_abilities = (BROKE_DENSE_ABILITY, GLORY_ABILITY)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a2.png"), main_color=(0, 200, 255), **kwargs):
        super(Rider, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

class Archer(RemoteSoldier):

    name = "特洛伊弓箭手"

    basic_walk_speed = 2  # px/每帧
    basic_run_speed = 3
    basic_close_combat_attack = 20
    basic_close_combat_defense = 22
    basic_charge_add = 0.01  # 冲锋加成

    basic_remote_combat_defense = 20
    basic_remote_combat_attack = 45
    basic_remote_reload_time = 20

    basic_remote_attack_dist_min = 64  # 最近射程
    basic_remote_attack_dist_max = 512  # 最远射程
    basic_remote_attack_range = 96  # 射击覆盖范围，越小越准
    basic_ammo_num = 120

    basic_morale = 10
    basic_soldier_nums = 400
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 20

    basic_abilities = [REMOTE_ATTACK_ABILITY, REMOTE_ATTACK_PRECISE]

    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "archer.png"), main_color=(0, 200, 255), **kwargs):
        super(Archer, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

    # 精确射击
    def _command_ability1(self):
        if not self.ability_enabled(REMOTE_ATTACK_ABILITY): # 未开启射击则不可控此项
            return
        if self.ability_enabled(REMOTE_ATTACK_PRECISE):
            self.current_abilities.remove(REMOTE_ATTACK_PRECISE)
        else:
            self.current_abilities.append(REMOTE_ATTACK_PRECISE)
        if self.remote_attack_combat != None:
            self.remote_attack_combat.reset_range(self.current_remote_attack_range())

class TroyGuard(Soldier):

    name = "特洛伊城邦护卫队"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 2
    basic_close_combat_attack = 62
    basic_close_combat_defense = 65
    basic_charge_add = 0.04  # 冲锋加成

    basic_remote_combat_defense = 70

    basic_morale = 11
    basic_soldier_nums = 250
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 15

    basic_abilities = (IN_DENSE, COURAGE_ABILITY, IMMUNE_THREAT, GLORY_ABILITY)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a16.png"), main_color=(0, 200, 255), **kwargs):
        super(TroyGuard, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

    # 密集阵型
    def _command_ability1(self):
        if self.ability_enabled(IN_DENSE):
            self.current_abilities.remove(IN_DENSE)
        else:
            self.current_abilities.append(IN_DENSE)

class Hector(Soldier):
    name = "赫克托尔精英军团"

    basic_walk_speed = 2  # px/每帧
    basic_run_speed = 4
    basic_close_combat_attack = 95
    basic_close_combat_defense = 100
    basic_charge_add = 0.15  # 冲锋加成

    basic_remote_combat_defense = 80

    basic_morale = 15
    basic_soldier_nums = 150
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 10

    basic_abilities = [COURAGE_ABILITY, BROKE_DENSE_ABILITY, COMMAND_ORDER, COMMAND_ABILITY, IMMUNE_THREAT, GLORY_ABILITY]
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a15.png"), main_color=(0, 64, 224), scale_size=(40, 58), **kwargs):
        super(Hector, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, scale_size=scale_size, **kwargs)

        self.cold_command_clock = COLD_CLOCK_ABILITY
        self.maintain_command_clock = 0

    # 时钟更新
    def preprocess_turn(self):
        if self.ability_enabled(COMMAND_ORDER):
            self.maintain_command_clock += 1
            if self.maintain_command_clock >= MAINTAIN_CLOCK_ABILITY:   # 到达技能保持时间
                self.current_abilities.remove(COMMAND_ORDER)
                self.maintain_command_clock = 0
        else:
            self.cold_command_clock += 1
            if self.cold_command_clock > COLD_CLOCK_ABILITY:
                self.cold_command_clock = COLD_CLOCK_ABILITY
        super(Hector, self).preprocess_turn()

    # 军事指挥
    def _command_ability1(self):
        if not self.ability_enabled(COMMAND_ORDER) and self.cold_command_clock == COLD_CLOCK_ABILITY:
            self.current_abilities.append(COMMAND_ORDER)
            self.cold_command_clock = 0

class Priam(Soldier):
    name = "特洛伊国王亲卫队"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 1
    basic_close_combat_attack = 65
    basic_close_combat_defense = 85
    basic_charge_add = 0.02  # 冲锋加成

    basic_remote_combat_defense = 90

    basic_morale = 15
    basic_soldier_nums = 500
    basic_strength = 12  # 体力
    basic_stay_from_battle_time = 10

    basic_abilities = (IN_DENSE, COURAGE_ABILITY, IMMUNE_THREAT, GLORY_ABILITY)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a3.png"), main_color=(0, 200, 224), **kwargs):
        super(Priam, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

    # 密集阵型
    def _command_ability1(self):
        if self.ability_enabled(IN_DENSE):
            self.current_abilities.remove(IN_DENSE)
        else:
            self.current_abilities.append(IN_DENSE)