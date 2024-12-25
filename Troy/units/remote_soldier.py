from typing import Any

from units.soldier import Soldier
from config_values import *
from utils.common import cal_distance


class RemoteSoldier(Soldier):

    basic_remote_combat_attack = 1
    basic_remote_reload_time = 20

    basic_remote_attack_dist_min = 64  # 最近射程
    basic_remote_attack_dist_max = MAX_RANGE  # 最远射程
    basic_remote_attack_range = 96  # 射击覆盖范围，越小越准
    basic_ammo_num = 100

    def __init__(self, *args, **kwargs):
        super(RemoteSoldier, self).__init__(*args, **kwargs)

        self.remote_attack_combat = None
        self.chasing_target_remote_attack = None

        self.current_remote_combat_attack = self.basic_remote_combat_attack
        self.current_remote_reload_time = self.basic_remote_reload_time
        self.current_ammo_num = self.basic_ammo_num

        self.remote_clock = self.current_remote_reload_time - 1

    def update(self, *args: Any, **kwargs: Any):
        # 射击检查
        self._update_remote_clock()
        if not self.is_surrending() and self.chasing_target_remote_attack != None:
            if not self.chasing_target_remote_attack in self.group.sprites():
                self.command_moving_state(0)    # 停止所有行动
                if self.remote_attack_combat != None:
                    self.group.remove_remote_combat(self.remote_attack_combat)
            else:
                if self.remote_attack_combat != None:
                    if self.remote_attack_combat.hit_area(self.chasing_target_remote_attack) > 0:   # 目标在射击战斗中（处于射击圆的区域）
                        return
                    else:   # 取消当前射击，开始追击敌军
                        self.current_abilities.remove(REMOTE_ATTACK_ABILITY)
                        self.group.remove_remote_combat(self.remote_attack_combat)
                wpos = self.chasing_target_remote_attack.world_position
                if self.check_in_remote_range(wpos):    # 这个是判断是否已进入射程
                    if not self.ability_enabled(REMOTE_ATTACK_ABILITY): # 未在射击则开启射击模式
                        self.current_abilities.append(REMOTE_ATTACK_ABILITY)
                    self.group.add_remote_combat(self, wpos, self.current_remote_attack_range())
                # 生成追击序列
                else:
                    if self.ability_enabled(REMOTE_ATTACK_ABILITY): # 取消射击状态
                        self.current_abilities.remove(REMOTE_ATTACK_ABILITY)
                    self.command_moving_state(1)  # 追击
                    self._get_target_position_path_sequence(wpos)
        super(RemoteSoldier, self).update()

    def _update_remote_clock(self):
        self.remote_clock += 1
        if self.remote_clock > self.current_remote_reload_time:
            self.remote_clock = self.current_remote_reload_time
    def _check_under_command(self):
        ratio = super(RemoteSoldier, self)._check_under_command()
        self.current_remote_combat_attack = self.basic_remote_combat_attack * ratio
        self.current_remote_reload_time = self.basic_remote_reload_time / ratio

    def current_remote_attack_range(self):
        if self.ability_enabled(REMOTE_ATTACK_PRECISE):
            return self.basic_remote_attack_range * REMOTE_PRECISE_RATIO
        else:
            return self.basic_remote_attack_range
    def current_remote_attack(self):    # 计算远程攻击时考虑自身人数
        return self.current_remote_combat_attack / (self.current_remote_attack_range()**2) * self.current_soldier_nums

    def check_in_remote_range(self, world_position):
        dis = cal_distance(world_position, self.world_position)
        if dis - self.basic_remote_attack_range / 2 >= self.basic_remote_attack_dist_min and dis + self.basic_remote_attack_range / 2 <= self.basic_remote_attack_dist_max:
            return True
        else:
            return False

    def set_in_close_combat(self, enemy, combat):
        super(RemoteSoldier, self).set_in_close_combat(enemy, combat)
        if self.ability_enabled(REMOTE_ATTACK_ABILITY): # 取消远程攻击能力
            self.current_abilities.remove(REMOTE_ATTACK_ABILITY)
            if self.in_remote_combat():
                self.group.remove_remote_combat(self.remote_attack_combat)
        if self.ability_enabled(REMOTE_ATTACK_PRECISE): # 取消精确射击
            self.current_abilities.remove(REMOTE_ATTACK_PRECISE)
    def set_in_remote_combat(self, combat):
        self.remote_attack_combat = combat
        self.current_moving_state = 0   # 站立射击
    def set_surrending(self):
        super(RemoteSoldier, self).set_surrending()
        if self.in_remote_combat():
            self.group.remove_remote_combat(self.remote_attack_combat)
    def set_ammo_exhausted(self):
        self._command_abilityq()
    def in_remote_combat(self):
        return self.remote_attack_combat is not None

    ###########
    def command_chasing_enemy(self, e):
        if self.is_surrending():
            return
        # 解除当前射击
        if self.in_remote_combat():
            self.group.remove_remote_combat(self.remote_attack_combat)
        if self.ability_enabled(REMOTE_ATTACK_ABILITY):
            self._command_chasing_enemy_remote_attack(e)
        else:
            super(RemoteSoldier, self).command_chasing_enemy(e)
            self.chasing_target_remote_attack = None
    def command_moving_position(self, world_position):
        if self.is_surrending():
            return
        # 如果是射击状态，视为尝试在对该点进行打击
        if self.ability_enabled(REMOTE_ATTACK_ABILITY):
            if self.check_in_remote_range(world_position):
                if self.in_remote_combat():
                    self.group.remove_remote_combat(self.remote_attack_combat)
                self.group.add_remote_combat(self, world_position, self.current_remote_attack_range())
        else:
            super(RemoteSoldier, self).command_moving_position(world_position)

    ####
    def _command_chasing_enemy_remote_attack(self, e):
        # 追击射击
        self.chasing_target_remote_attack = e
        self.chasing_target = None
    def _command_abilityq(self):
        self.chasing_target_remote_attack = None
        self.chasing_target = None
        self.command_moving_state(0)
        # 若在战斗中，无法切换状态
        if self.in_close_combat():
            return
        # 切换射击和战斗
        if self.ability_enabled(REMOTE_ATTACK_ABILITY):
            self.current_abilities.remove(REMOTE_ATTACK_ABILITY)
            if self.in_remote_combat():
                self.group.remove_remote_combat(self.remote_attack_combat)
        elif self.current_ammo_num > 0: # 仅当此条件满足时允许开启
            self.current_abilities.append(REMOTE_ATTACK_ABILITY)


