from config_values import *
from utils.common import set_in_range


class MoraleManager():

    MORALE_DICT = {
        0: ("战斗中持续伤亡", -0.072),
        1: ("腹背受敌", -0.072),
        2: ("四面受敌", -0.096),
        3: ("寡不敌众", -0.096),
        4: ("尝试脱离战斗", -0.008),
        5: ("疲惫不堪", -0.016),
        6: ("筋疲力尽", -0.032),
        7: ("国王阵亡", -0.048),
        8: ("冲锋中", 0.064),
        9: ("战斗勇气", 0.072),
        10: ("受到附近部队激励", 0.048),
        11: ("受到敌军恐吓", -0.024),
        12: ("追杀敌军", 0.036),
        13: ("英雄陨落", -0.064),
        14: ("友军溃逃", -0.048),
        15: ("己方人数占优", 0.032),
        16: ("敌军溃逃", 0.096)
    }

    def __init__(self):
        self._morale_dict = dict()

    def _remove(self, item):
        if item[0] in self._morale_dict:
            self._morale_dict.pop(item[0])

    def check_continous_hurt(self, hurt_ratio, last_in_battle_clock, current_clock, basic_stay_from_battle_time):
        item0 = MoraleManager.MORALE_DICT[0]
        item1 = MoraleManager.MORALE_DICT[9]
        # 战斗中持续伤亡
        if current_clock - last_in_battle_clock < basic_stay_from_battle_time and hurt_ratio < 0.9:  # 伤亡
            self._remove(item1)
            self._morale_dict[item0[0]] = item0[1] * (1- hurt_ratio)
        else:
            self._remove(item0)
            # 脱离战斗的时间越长，越能捡回战斗勇气
            self._morale_dict[item1[0]] = set_in_range(item1[1] * ((current_clock - last_in_battle_clock) / (basic_stay_from_battle_time*4))**(1/2), item1[1], 2 * item1[1])

    def check_charge(self, current_moving_state):
        item = MoraleManager.MORALE_DICT[8]
        if current_moving_state == 3:
            self._morale_dict[item[0]] = item[1]
        else:
            self._remove(item)

    def check_arround(self, this_s):
        item0 = MoraleManager.MORALE_DICT[3]    # 寡不敌众
        item1 = MoraleManager.MORALE_DICT[14]   # 友军溃逃
        item2 = MoraleManager.MORALE_DICT[15]   # 己方人数占优
        item3 = MoraleManager.MORALE_DICT[16]   # 敌军溃逃
        # 周围溃逃友军的人数大于未溃逃友军的人数，则触发友军溃逃
        sf_num = 0
        ef_num = 0
        s_num = 0
        e_num = 0
        for s in this_s.distance_dict["arround_friendly"]:
            if s.is_surrending():
                sf_num += 1
            else:
                sf_num -= 1
            s_num += s.current_soldier_nums
        for e in this_s.distance_dict["arround_enemy"]:
            # 统计范围内情况
            if e.is_surrending():
                ef_num += 1
            else:
                ef_num -= 1
            e_num += e.current_soldier_nums
        # 崩溃影响
        if sf_num > 0:
            self._morale_dict[item1[0]] = item1[1]
        else:
            self._remove(item1)
        if ef_num > 3:
            self._morale_dict[item3[0]] = item3[1]
        else:
            self._remove(item3)
        # 面对敌人10x于己，触发寡不敌众
        if not this_s.ability_enabled(GLORY_ABILITY):
            if e_num / (s_num + this_s.current_soldier_nums) >= 10:
                self._morale_dict[item0[0]] = item0[1]
            else:
                self._remove(item0)
        # 己方人数占优
        if (e_num > 0 and s_num / e_num > 2) or (e_num == 0 and s_num / this_s.current_soldier_nums >= 10):
            self._morale_dict[item2[0]] = item2[1]
        else:
            self._remove(item2)

    def check_fight(self, in_besieged, in_pinched):
        # 有一个既有1也既有-1，另外一个不全为一个方向，则为两面受敌
        item0 = MoraleManager.MORALE_DICT[2]
        item1 = MoraleManager.MORALE_DICT[1]
        if in_besieged:
            self._morale_dict[item0[0]] = item0[1]
            # 取消被夹击
            self._remove(item1)
        else:
            if in_pinched:
                self._morale_dict[item1[0]] = item1[1]
            else:
                self._remove(item1)
            self._remove(item0)

    def check_moving_from_combat(self, moving_from_combat, in_actual_combat):
        # 脱离战斗
        item = MoraleManager.MORALE_DICT[4]
        if moving_from_combat and in_actual_combat:
            self._morale_dict[item[0]] = item[1]
        else:
            self._remove(item)

    def check_strength(self, current_strength, level4, level3):
        # 体力变化
        item0 = MoraleManager.MORALE_DICT[5]
        item1 = MoraleManager.MORALE_DICT[6]
        if current_strength < level4:
            self._remove(item0)
            self._morale_dict[item1[0]] = item1[1]
        elif current_strength < level3:
            self._remove(item1)
            self._morale_dict[item0[0]] = item0[1]
        else:
            self._remove(item0)
            self._remove(item1)

    def check_king_died(self, king, ratio=1):
        # 国王阵亡
        item = MoraleManager.MORALE_DICT[7]
        if king == None:
            self._morale_dict[item[0]] = item[1] * ratio # 永不可删除

    def check_courage(self, courage_list):
        # 随距离变化的激励效果
        item = MoraleManager.MORALE_DICT[10]
        courage_value = 0
        for _, dis in courage_list:
            courage_value += 2*(1 - dis / EFFECTIVE_COURAGE_RANGE) * item[1]
        if courage_value > 0:
            self._morale_dict[item[0]] = set_in_range(courage_value, 0, 2 * item[1])
        else:
            self._remove(item)

    def check_threatened(self, threat_list):
        item = MoraleManager.MORALE_DICT[11]
        threat_value = 0
        for _, dis in threat_list:
            threat_value += 2 * (1 - dis / EFFECTIVE_THREATEN_RANGE) * item[1]
        if threat_value < 0:
            self._morale_dict[item[0]] = set_in_range(threat_value, 0, 2 * item[1])
        else:
            self._remove(item)

    def check_chasing_flee(self, in_actual_combat, joining_combats):
        item = MoraleManager.MORALE_DICT[12]
        if not in_actual_combat and len(joining_combats) > 0:
            self._morale_dict[item[0]] = item[1]
        else:
            self._remove(item)

    def check_hero_dead(self, hero, ratio=1):
        item = MoraleManager.MORALE_DICT[13]
        if hero == None:
            self._morale_dict[item[0]] = item[1] * ratio

    def get_infos(self):
        return list(self._morale_dict.keys())

    def get_values(self):
        return list(self._morale_dict.values())

    def get_items(self):
        return self._morale_dict.items()

    def clear(self):
        self._morale_dict.clear()