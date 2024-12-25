import time
from typing import List, Union, Any

import pygame.sprite
from pygame import Rect
from threading import Thread

from managers.combat_manager import *
from managers.soldier_manager import SoldierManager
from units.remote_soldier import RemoteSoldier
from units.soldier import Soldier
from utils.common import *
from config_assert import *
from config_values import *

class InfoSurface(pygame.surface.Surface):
    
    def __init__(self, *args, sprite=None, fillValue=(21, 21, 21)):
        super(InfoSurface, self).__init__(*args)
        self.fill(fillValue)

        # 标题
        font_name = pygame.font.SysFont('simHei', 22)
        self.blit(font_name.render(sprite.name, True, sprite.main_color), (20, 10))

        self.y_last = 10
        self.x_begin = 5
        self.rowHeight = 30
        self.content_distance = 50
        self.font_info = pygame.font.SysFont('simHei', 18)

    def add_row(self, title=None, content=None, content_color=(255, 255, 255), content_offset=0):
        if title != None:
            self.blit(self.font_info.render(f"{title}：", True, (255, 255, 255)), (self.x_begin, self.y_last+self.rowHeight))
        if content != None:
            self.blit(self.font_info.render(content, True, content_color), (self.x_begin+self.content_distance+content_offset, self.y_last+self.rowHeight))

        self.y_last = self.y_last + self.rowHeight

class MainGroup(pygame.sprite.LayeredUpdates):
    
    def __init__(self, map_layer=None, screen_size=None):
        super(MainGroup, self).__init__()
        # 地图
        self._map_layer = map_layer
        # 显示相关
        self.screen_size = screen_size
        if not screen_size == None:
            self.info_width = 280
            self.info_location = (screen_size[0], 0)
            self.main_surface = pygame.surface.Surface(self.screen_size)
            # 显示超参数
            self.border_width = 3
            self.bar_width = 5
            self.morale_bar_height = 20
            self.num_bar_height = 30
            # 显示资源
            self.move_ = pygame.transform.scale(pygame.image.load(MOVE_PNG), (20, 20))
            self.move2x_ = pygame.transform.scale(pygame.image.load(MOVE_2X_PNG), (20, 20))
            self.charge_ = pygame.transform.scale(pygame.image.load(CHARGE_PNG), (20, 20))
            self.aim_ = pygame.transform.scale(pygame.image.load(AIM_PNG), (20, 20))
            self.target_ = pygame.transform.scale(pygame.image.load(TARGET_PNG), (20, 20))
            self.moving_icon_offset = (0, 30)
            self.shield_ = pygame.transform.scale(pygame.image.load(SHIELD_PNG), (30, 30))
            self.gold_shield_ = pygame.transform.scale(pygame.image.load(GOLD_SHIELD_PNG), (30, 30))
            self.arrow_ = pygame.transform.scale(pygame.image.load(ARROW_PNG), (30, 30))
            self.sword_ = pygame.transform.scale(pygame.image.load(SWORD_PNG), (30, 30))
            self.ability1_icon_offset = (20, 20)
            self.set_arrows_ = pygame.image.load(SET_ARROWS)  # 远程攻击位置
            self.command_ = pygame.transform.scale(pygame.image.load(COMMAND_PNG), (20, 20))
            self.ability2_icon_offset = (0, 0)
        # 管理器
        self.combats = set()
        self.soldiers = set()
        self.soldier_manager = SoldierManager()
        self.current_choose_s = None
        self.need_flush_this_turn_ = set()
        # 全局同步时钟
        self.clock = 0
        self.game_exit = False

    def screen_to_world(self, screen_x, screen_y):
        offset_x, offset_y = self._map_layer.get_center_offset()
        world_x = screen_x / self._map_layer.zoom - offset_x
        world_y = screen_y / self._map_layer.zoom - offset_y
        return int(world_x), int(world_y)

    def set_choose(self, sprite=None):
        self.current_choose_s = sprite
        # 设置选中框，在视窗右侧展示详细信息
        if sprite != None:
            # 在视窗右侧展示
            pygame.display.set_mode((self.screen_size[0] + self.info_width, self.screen_size[1]))
        else:
            pygame.display.set_mode(self.screen_size)

    def add_combat(self, s1, s2, world_position):
        if s1.troop_id == 0:
            combat = Combat(world_position, [s1, s2])
        else:
            combat = Combat(world_position, [s2, s1])
        # 计算首战杀伤
        s1.propose_soldier_nums_encounter(s2, world_position)
        s2.propose_soldier_nums_encounter(s1, world_position)
        #
        s1.set_in_close_combat(s2, combat)
        s2.set_in_close_combat(s1, combat)
        # 算法优化
        self.need_flush_this_turn_.add(s1)
        self.need_flush_this_turn_.add(s2)
        #
        self.combats.add(combat)
        self.add(combat)
    def remove_combat(self, combat):
        combat.s[0].joining_combates.remove(combat)
        combat.s[0].fighting_enemies.remove(combat.s[1])
        combat.s[1].joining_combates.remove(combat)
        combat.s[1].fighting_enemies.remove(combat.s[0])
        self.combats.remove(combat)
        self.remove(combat)
    def add_remote_combat(self, archer, position, attack_range):
        if archer.remote_attack_combat != None:
            self.remove_remote_combat(archer.remote_attack_combat)
        combat = RemoteCombat(archer, position, attack_range)
        archer.set_in_remote_combat(combat)
        self.combats.add(combat)
        self.add(combat)
    def remove_remote_combat(self, combat):
        combat.archer.remote_attack_combat = None
        self.combats.remove(combat)
        self.remove(combat)
    def add_soldier(self, soldier, which_troop, is_king, is_hero):
        self.soldier_manager.add_soldier(soldier, which_troop, is_king, is_hero)
        self.soldiers.add(soldier)
        self.add(soldier)
    def remove_soldier(self, soldier):
        # 解除所有关联的战斗
        for combat in soldier.joining_combates.copy():
            self.remove_combat(combat)
        if REMOTE_ATTACK_ABILITY in soldier.basic_abilities and soldier.remote_attack_combat != None:
            self.remove_remote_combat(soldier.remote_attack_combat)
        # 删除
        self.soldier_manager.remove_soldier(soldier, clock=self.clock)
        self.soldiers.remove(soldier)
        self.remove(soldier)
        if self.current_choose_s == soldier:
            self.set_choose()
        print(f"{soldier} has been killed.")

    def update(self, *args: Any, **kwargs: Any) -> None:
        # 初始化
        self.clock += 1 # 更新时钟
        self.soldier_manager.set_all_kh_ratio(clock=self.clock)
        self.need_flush_this_turn_ = set()
        # 预处理和划分
        for sold in self.soldiers:
            sold.pppreprocess_turn() # 重置距离字典
        for sold in self.soldiers:
            sold.ppreprocess_turn() # 记录相互间距离
        for sold in self.soldiers:
            sold.preprocess_turn()
        # 战斗列表更新（提议所有的变化，但不马上更新状态）
        # （作战伤亡，包括远程和近战伤亡）->士气变化、体力变化、人数变化->（人数更新）->位置变化->（新战斗触发、首战杀伤、人数变化）
        for combat in self.combats.copy():
            combat.update(self.soldiers, *args, **kwargs)
        for sold in self.soldiers:
            sold.propose()  # 士气提议、体力提议、人数提议
        for sold in self.soldiers.copy():
            sold.flush_all() # 结算所有变化，可能会导致移动的变化
        # 更新其余状态（多线程
        for sold in self.soldiers.copy():
            sold.update()
        # 新战斗的首战杀伤更新和结束回合
        for sprite in self.need_flush_this_turn_:
            sprite.flush_number()

    def draw(
            self,
            surface: pygame.surface.Surface
    ):    # 所有的绘制都要考虑到放缩的因素
        """
        Draw map and all sprites onto the surface.

        Args:
            surface: Surface to draw to

        """
        # 将内容显示在一个surface上
        screen = surface
        surface = self.main_surface
        surface.fill((0, 0, 0))

        ox, oy = self._map_layer.get_center_offset()
        draw_area = surface.get_rect()
        view_rect = self._map_layer.view_rect.copy()    # 展示位置
        zoom = self._map_layer.zoom

        new_surfaces = list()
        spritedict = self.spritedict
        gl = self.get_layer_of_sprite
        new_surfaces_append = new_surfaces.append

        # 选中对象的设置
        gccs = self.current_choose_s
        if gccs != None:
            # 若具有激励能力
            if gccs.ability_enabled(COURAGE_ABILITY):
                circle_surface_cou = pygame.surface.Surface((EFFECTIVE_COURAGE_RANGE*2, EFFECTIVE_COURAGE_RANGE*2), pygame.SRCALPHA)
                pygame.draw.circle(circle_surface_cou, (0, 64, 200, 96), circle_surface_cou.get_rect().center, EFFECTIVE_COURAGE_RANGE)
                # 获取屏幕位置
                new_surfaces_append((circle_surface_cou, circle_surface_cou.get_rect().move(gccs.world_position).move(ox, oy).move(-EFFECTIVE_COURAGE_RANGE, -EFFECTIVE_COURAGE_RANGE),
                                     gl(gccs)))
            # 若具有指挥能力
            if gccs.ability_enabled(COMMAND_ABILITY):
                circle_surface_com = pygame.surface.Surface((EFFECTIVE_COMMAND_RANGE * 2, EFFECTIVE_COMMAND_RANGE * 2),
                                                        pygame.SRCALPHA)
                pygame.draw.circle(circle_surface_com, (0, 200, 200, 96), circle_surface_com.get_rect().center,
                                   EFFECTIVE_COMMAND_RANGE)
                # 获取屏幕位置
                new_surfaces_append((circle_surface_com,
                                     circle_surface_com.get_rect().move(gccs.world_position).move(ox, oy).move(
                                         -EFFECTIVE_COMMAND_RANGE, -EFFECTIVE_COMMAND_RANGE),
                                     gl(gccs)))
            # 若是弓箭手
            if gccs.ability_enabled(REMOTE_ATTACK_ABILITY):
                mouse_pos = pygame.mouse.get_pos()
                mouse_world_pos = self.screen_to_world(*mouse_pos)
                # 射程圆
                circle_surface = pygame.surface.Surface((gccs.basic_remote_attack_dist_max * 2 - gccs.basic_remote_attack_range,
                                                         gccs.basic_remote_attack_dist_max * 2 - gccs.basic_remote_attack_range), pygame.SRCALPHA)
                pygame.draw.circle(circle_surface, (200, 64, 0, 96), (circle_surface.get_rect().center),
                                   gccs.basic_remote_attack_dist_max - gccs.basic_remote_attack_range / 2)
                pygame.draw.circle(circle_surface, (0, 0, 0, 0), (circle_surface.get_rect().center),
                                   gccs.basic_remote_attack_dist_min + gccs.basic_remote_attack_range / 2)
                # 获取屏幕位置
                new_surfaces_append(
                    (circle_surface, circle_surface.get_rect().move(gccs.world_position).move(ox, oy).move(
                        -gccs.basic_remote_attack_dist_max + gccs.basic_remote_attack_range / 2, -gccs.basic_remote_attack_dist_max + gccs.basic_remote_attack_range / 2),
                     gl(gccs)))
                #  在鼠标处显示靶环
                if gccs.check_in_remote_range(mouse_world_pos):
                    r = gccs.current_remote_attack_range()
                    img = pygame.transform.scale(self.set_arrows_, (r, r))
                    new_surfaces_append((img, img.get_rect().move(*mouse_world_pos).move(ox, oy).move(-r//2, -r//2), gl(gccs)))

        bars = []
        # 添加所有精灵
        for spr in self.sprites():  # 在窗口内展示

            spr_rect = spr.rect # 世界坐标
            new_rect = spr_rect.move(ox, oy)    # 转屏幕坐标
            if spr_rect.colliderect(view_rect):
                if isinstance(spr, Soldier):    # 条处理
                    # showing check
                    if not spr.is_show_update():
                        continue
                    # 添加状态图标
                    moving_state_icon_rect = new_rect.move(self.moving_icon_offset)
                    if spr.current_moving_state == 0:
                        if spr.ability_enabled(REMOTE_ATTACK_ABILITY):
                            if spr.remote_attack_combat != None:
                                new_surfaces_append((self.target_, moving_state_icon_rect, gl(spr)+1))
                            else:
                                new_surfaces_append((self.aim_, moving_state_icon_rect, gl(spr)+1))
                    elif spr.current_moving_state == 1:
                        new_surfaces_append((self.move_, moving_state_icon_rect, gl(spr)+1))
                    elif spr.current_moving_state == 2:
                        new_surfaces_append((self.move2x_, moving_state_icon_rect, gl(spr)+1))
                    elif spr.current_moving_state == 3:
                        new_surfaces_append((self.charge_, moving_state_icon_rect, gl(spr)+1))
                    # 其余状态
                    ability1_icon_rect = new_rect.move(self.ability1_icon_offset)
                    if spr.ability_enabled(IN_DENSE):
                        new_surfaces_append((self.shield_, ability1_icon_rect, gl(spr)+1))
                    if spr.ability_enabled(IN_TESTUDO):
                        new_surfaces_append((self.gold_shield_, ability1_icon_rect, gl(spr)+1))
                    if spr.ability_enabled(STRIKE_ATTACK):
                        new_surfaces_append((self.sword_, ability1_icon_rect, gl(spr)+1))
                    #
                    ability2_icon_rect = new_rect.move(self.ability2_icon_offset)
                    if spr.current_under_command == 2:
                        new_surfaces_append((self.command_, ability2_icon_rect, gl(spr)+1))
                    if spr.current_hurt_by_remote:
                        new_surfaces_append((self.arrow_, ability2_icon_rect, gl(spr)+1))
                    # 添加士气条和人数条
                    # 计算当前条长度
                    current_nums_height = int((spr.current_soldier_nums / spr.basic_soldier_nums) * self.num_bar_height)    # 绝对值
                    current_morale_height = int((spr.current_morale / MORALE_DROP_STANDARD) * self.morale_bar_height)
                    # 绘制当前人数的人数条
                    if spr_rect[0] - 2*self.bar_width >= view_rect[0]:   # 左边展示空间足够
                        bars.append([(screen, get_diff_color_level(spr.current_soldier_nums / spr.basic_soldier_nums),
                                      self._map_layer.translate_rect(Rect(
                                          (spr_rect[0]-self.bar_width, spr_rect[1], self.bar_width, current_nums_height)))),
                                   (screen, get_diff_color_level(spr.current_morale, [
                                       MORALE_DROP_LEVEL1, MORALE_DROP_LEVEL2, MORALE_DROP_LEVEL3
                                   ]), self._map_layer.translate_rect(Rect((spr_rect[0] - 2 * self.bar_width), spr_rect[1],
                                                                           self.bar_width, current_morale_height)))])
                    else:
                        bars.append([(screen, get_diff_color_level(spr.current_soldier_nums / spr.basic_soldier_nums),
                                      self._map_layer.translate_rect(Rect((spr_rect[0] + spr_rect[2], spr_rect[1], self.bar_width, current_nums_height)))),
                                     (screen, get_diff_color_level(spr.current_morale, [
                                       MORALE_DROP_LEVEL1, MORALE_DROP_LEVEL2, MORALE_DROP_LEVEL3
                                   ]), self._map_layer.translate_rect(Rect((spr_rect[0] + spr_rect[2] + self.bar_width), spr_rect[1], self.bar_width, current_morale_height)))])

                # 加入到界面中
                try:
                    new_surfaces_append((spr.image, new_rect, gl(spr), spr.blendmode))
                except AttributeError:
                    # should only fail when no blendmode available
                    new_surfaces_append((spr.image, new_rect, gl(spr)))
                spritedict[spr] = new_rect

        # 绘制一切
        self.lostsprites = []
        self._map_layer.draw(surface, draw_area, new_surfaces)
        screen.blit(surface, draw_area)

        # 添加条
        for ab, bb in bars:
            pygame.draw.rect(*ab)
            pygame.draw.rect(*bb)

        if gccs != None:
            # 添加信息
            image_rect = gccs.rect # 世界坐标
            border_rect = pygame.Rect(image_rect.left - self.border_width,
                                      image_rect.top - self.border_width,
                                      image_rect.width + 2 * self.border_width,
                                      image_rect.height + 2 * self.border_width)
            border_rect = self._map_layer.translate_rect(border_rect)   # 世界转屏幕
            pygame.draw.rect(screen, (0, 0, 0), border_rect, int(self.border_width * zoom))
            # 若在攻击状态，还要框出他的攻击区域
            if gccs.ability_enabled(REMOTE_ATTACK_ABILITY) and gccs.in_remote_combat():
                image_rect = gccs.remote_attack_combat.rect  # 世界坐标
                border_rect = pygame.Rect(image_rect.left - self.border_width,
                                          image_rect.top - self.border_width,
                                          image_rect.width + 2 * self.border_width,
                                          image_rect.height + 2 * self.border_width)
                border_rect = self._map_layer.translate_rect(border_rect)
                pygame.draw.rect(screen, (255, 255, 255), border_rect, int(self.border_width * zoom))
            # 展示信息
            info_surface = InfoSurface((self.info_width, self.screen_size[1]), sprite=gccs)
            info_surface.add_row("人数", f"{int(gccs.current_soldier_nums)}/{int(gccs.basic_soldier_nums)}",
                                 get_diff_color_level(gccs.current_soldier_nums / gccs.basic_soldier_nums))
            # 士气
            info_surface.add_row("士气", get_diff_level(gccs.current_morale,
                                                             [MORALE_DROP_LEVEL1, MORALE_DROP_LEVEL2,
                                                              MORALE_DROP_LEVEL3, MORALE_DROP_LEVEL4]),
                                 get_diff_color_level(gccs.current_morale,
                                                      [MORALE_DROP_LEVEL1, MORALE_DROP_LEVEL2, MORALE_DROP_LEVEL3]))
            if isinstance(gccs, RemoteSoldier):
                info_surface.add_row("弹药", get_diff_level(gccs.current_ammo_num/gccs.basic_ammo_num, [0.8, 0.45, 0.25, 0], ["充足", "足够", "尚可", "将尽", "耗尽"]),
                                     get_diff_color_level(gccs.current_ammo_num/gccs.basic_ammo_num, [0.75, 0.25, 0]))
            # 状态
            if not gccs.in_close_combat():
                if gccs.ability_enabled(REMOTE_ATTACK_ABILITY) and gccs.in_remote_combat():
                    state = "远程攻击中"
                    stateColor = (0, 255, 255)
                else:
                    state = "未加入战斗；"
                    stateColor = (0, 255, 0)
                    if gccs.current_moving_state == 0:
                        state = state + "站立"
                    elif gccs.current_moving_state == 1:
                        state = state + "行走中"
                    elif gccs.current_moving_state == 2:
                        state = state + "奔跑中"
                    elif gccs.current_moving_state == 3:
                        state = state + "冲锋中"
            else:
                state = "战斗中"
                stateColor = (255, 0, 0)
            if gccs.current_flee_state == 1:
                info_surface.add_row(content="逃亡中", content_color=(255, 0, 0))
            elif gccs.current_flee_state == 2:
                info_surface.add_row(content="已被击溃", content_color=(255, 255, 255))
            info_surface.add_row("状态", state, stateColor)
            if gccs.current_hurt_by_remote:
                info_surface.add_row(content="受到远程打击", content_color=(255, 255, 255))
            if gccs.current_under_command == 1:
                info_surface.add_row(content="受到将军指挥", content_color=(0, 255, 0))
            elif gccs.current_under_command == 2:
                info_surface.add_row(content="受到将军全力指挥", content_color=(0, 255, 0))
            # 特殊技能
            for ability in gccs.current_abilities:
                if ability[-1] == 1:  # 主动技能
                    info_surface.add_row(content=ability[0], content_color=(0, 255, 0), content_offset=-10)
                elif ability[-1] == 2:
                    info_surface.add_row(content=ability[0]+f"(剩余{MAINTAIN_CLOCK_ABILITY-gccs.maintain_command_clock}f)",
                                         content_color=(0, 255, 0))
            if COMMAND_ORDER in gccs.basic_abilities and not gccs.ability_enabled(COMMAND_ORDER):
                left_time = COLD_CLOCK_ABILITY-gccs.cold_command_clock
                info_surface.add_row(content=COMMAND_ORDER[0]+f"(冷却中，剩余{left_time}f)",
                                     content_color=(255, 0, 0) if left_time != 0 else (255, 255, 255))
            # 体力
            strlevel = gccs.get_strength_level()
            if strlevel == 0:
                info_surface.add_row("体力", "精力充沛", (0, 255, 0))
            elif strlevel == 1:
                info_surface.add_row("体力", "气喘吁吁", (64, 224, 0))
            elif strlevel == 2:
                info_surface.add_row("体力", "疲惫", (224, 180, 0))
            elif strlevel == 3:
                info_surface.add_row("体力", "疲惫不堪", (255, 64, 0))
            else:
                info_surface.add_row("体力", "筋疲力尽", (255, 0, 0))
            # 士气影响
            info_surface.add_row("士气影响")
            for morale_info, value in gccs.morale_set.get_items():
                info_surface.add_row(content=morale_info,
                                     content_color=(0, 255, 0) if value > 0 else (255, 0, 0),
                                     content_offset=-10)
            info_surface = info_surface
            screen.blit(info_surface, self.info_location)