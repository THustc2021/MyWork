import os.path

from units.remote_soldier import RemoteSoldier
from units.soldier import Soldier
from config_values import *

class MycenaeanSworder(Soldier):

    name = "迈锡尼持盾剑士"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 2
    basic_close_combat_attack = 46
    basic_close_combat_defense = 38
    basic_charge_add = 0.02  # 冲锋加成

    basic_remote_combat_defense = 75

    basic_morale = 10
    basic_soldier_nums = 750
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 30

    basic_abilities = (IN_DENSE, )
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a11.png"), main_color=(96, 96, 96, 255), **kwargs):
        super(MycenaeanSworder, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

    # 密集阵型
    def _command_ability1(self):
        if self.ability_enabled(IN_DENSE):
            self.current_abilities.remove(IN_DENSE)
        else:
            self.current_abilities.append(IN_DENSE)

class SpartarMan(Soldier):

    name = "斯巴达重装步兵"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 2
    basic_close_combat_attack = 35
    basic_close_combat_defense = 52
    basic_charge_add = 0.03  # 冲锋加成

    basic_remote_combat_defense = 60

    basic_morale = 10
    basic_soldier_nums = 500
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 20

    basic_abilities = (IN_TESTUDO, IMMUNE_THREAT)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a1.png"), main_color=(96, 96, 96), scale_size=(40, 62), **kwargs):
        super(SpartarMan, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, scale_size=scale_size, **kwargs)

    # 密集阵型
    def _command_ability1(self):
        if self.ability_enabled(IN_TESTUDO):
            self.current_abilities.remove(IN_TESTUDO)
        else:
            self.current_abilities.append(IN_TESTUDO)

class AthensMan(Soldier):

    name = "雅典军团"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 2
    basic_close_combat_attack = 48
    basic_close_combat_defense = 42
    basic_charge_add = 0.03  # 冲锋加成

    basic_remote_combat_defense = 45

    basic_morale = 10
    basic_soldier_nums = 750
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 20

    basic_abilities = (IN_DENSE, GLORY_ABILITY)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a7.png"), main_color=(96, 96, 96), **kwargs):
        super(AthensMan, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

    # 密集阵型
    def _command_ability1(self):
        if self.ability_enabled(IN_DENSE):
            self.current_abilities.remove(IN_DENSE)
        else:
            self.current_abilities.append(IN_DENSE)

class PylosMan(Soldier):
    name = "皮洛斯步兵"

    basic_walk_speed = 2  # px/每帧
    basic_run_speed = 3
    basic_close_combat_attack = 38
    basic_close_combat_defense = 30
    basic_charge_add = 0.1  # 冲锋加成

    basic_remote_combat_defense = 30

    basic_morale = 10
    basic_soldier_nums = 750
    basic_strength = 12  # 体力
    basic_stay_from_battle_time = 20
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a5.png"), main_color=(96, 96, 96), scale_size=(57, 40), **kwargs):
        super(PylosMan, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, scale_size=scale_size, **kwargs)

class MopoliaArcher(RemoteSoldier):

    name = "墨波利亚弓箭手"

    basic_walk_speed = 2  # px/每帧
    basic_run_speed = 3
    basic_close_combat_attack = 20
    basic_close_combat_defense = 26
    basic_charge_add = 0.01  # 冲锋加成

    basic_remote_combat_defense = 32
    basic_remote_combat_attack = 40
    basic_remote_reload_time = 10

    basic_remote_attack_dist_min = 64  # 最近射程
    basic_remote_attack_dist_max = 384  # 最远射程
    basic_remote_attack_range = 128  # 射击覆盖范围，越小越准
    basic_ammo_num = 100

    basic_morale = 8
    basic_soldier_nums = 500
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 20

    basic_abilities = [REMOTE_ATTACK_ABILITY, REMOTE_ATTACK_PRECISE]

    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "archer2.png"), main_color=(96, 96, 96), **kwargs):
        super(MopoliaArcher, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

    def _command_ability1(self):
        if not self.ability_enabled(REMOTE_ATTACK_ABILITY): # 未开启射击则不可控此项
            return
        if self.ability_enabled(REMOTE_ATTACK_PRECISE):
            self.current_abilities.remove(REMOTE_ATTACK_PRECISE)
        else:
            self.current_abilities.append(REMOTE_ATTACK_PRECISE)
        if self.remote_attack_combat != None:
            self.remote_attack_combat.reset_range(self.current_remote_attack_range())

class GreeceArcher(RemoteSoldier):

    name = "希腊弓箭手"

    basic_walk_speed = 2  # px/每帧
    basic_run_speed = 3
    basic_close_combat_attack = 28
    basic_close_combat_defense = 25
    basic_charge_add = 0.01  # 冲锋加成

    basic_remote_combat_defense = 30
    basic_remote_combat_attack = 30
    basic_remote_reload_time = 10

    basic_remote_attack_dist_min = 64  # 最近射程
    basic_remote_attack_dist_max = 512  # 最远射程
    basic_remote_attack_range = 256  # 射击覆盖范围，越小越准
    basic_ammo_num = 80
    basic_morale = 8
    basic_soldier_nums = 750
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 30

    basic_abilities = [REMOTE_ATTACK_ABILITY]

    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "archer.png"), main_color=(96, 96, 96), **kwargs):
        super(GreeceArcher, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)

class ArgosMan(Soldier):
    name = "阿尔戈斯军团"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 1
    basic_close_combat_attack = 65
    basic_close_combat_defense = 50
    basic_charge_add = 0.02  # 冲锋加成

    basic_remote_combat_defense = 75

    basic_morale = 10
    basic_soldier_nums = 750
    basic_strength = 10  # 体力
    basic_stay_from_battle_time = 15

    basic_abilities = (IN_TESTUDO, IMMUNE_THREAT)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a20.png"), main_color=(96, 96, 96), scale_size=(30, 70), **kwargs):
        super(ArgosMan, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, scale_size=scale_size, **kwargs)

    def _command_ability1(self):
        if self.ability_enabled(IN_TESTUDO):
            self.current_abilities.remove(IN_TESTUDO)
        else:
            self.current_abilities.append(IN_TESTUDO)

class Achilles(Soldier):

    name = "阿基里斯战团"

    basic_walk_speed = 1  # px/每帧
    basic_run_speed = 3
    basic_close_combat_attack = 150
    basic_close_combat_defense = 150
    basic_charge_add = 0.08  # 冲锋加成

    basic_remote_combat_defense = 120

    basic_morale = 13
    basic_soldier_nums = 100
    basic_strength = 12  # 体力
    basic_stay_from_battle_time = 10

    basic_abilities = (IN_TESTUDO, STRIKE_ATTACK, COURAGE_ABILITY, THREATEN_ENEMY_ABILITY, IMMUNE_THREAT, GLORY_ABILITY)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a17.png"), main_color=(124, 64, 96), scale_size=(40, 67), **kwargs):
        super(Achilles, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, scale_size=scale_size, **kwargs)

    def _command_ability1(self):
        if self.ability_enabled(IN_TESTUDO):
            self.current_abilities.remove(IN_TESTUDO)
        else:
            if self.ability_enabled(STRIKE_ATTACK):
                self.current_abilities.remove(STRIKE_ATTACK)
            self.current_abilities.append(IN_TESTUDO)

    def _command_ability2(self):
        if self.ability_enabled(STRIKE_ATTACK):
            self.current_abilities.remove(STRIKE_ATTACK)
        else:
            if self.ability_enabled(IN_TESTUDO):
                self.current_abilities.remove(IN_TESTUDO)
            self.current_abilities.append(STRIKE_ATTACK)

class AkaMenon(Soldier):

    name = "阿伽门农国王卫队"

    basic_walk_speed = 2  # px/每帧
    basic_run_speed = 3
    basic_close_combat_attack = 85
    basic_close_combat_defense = 100
    basic_charge_add = 0.08  # 冲锋加成

    basic_remote_combat_defense = 85

    basic_morale = 14
    basic_soldier_nums = 30
    basic_strength = 12  # 体力
    basic_stay_from_battle_time = 5

    basic_abilities = (COURAGE_ABILITY, IMMUNE_THREAT, GLORY_ABILITY)
    def __init__(self, *args, portrait_path=os.path.join("fix_soldiers", "a0.png"), main_color=(124, 96, 96), **kwargs):
        super(AkaMenon, self).__init__(*args, portrait_path=portrait_path, main_color=main_color, **kwargs)
