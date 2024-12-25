from typing import Any

import pygame.image

from config_assert import COMBAT_PNG, ARROWS_TARGET


class Combat(pygame.sprite.Sprite):

    image = pygame.transform.scale(pygame.image.load(COMBAT_PNG), (35, 30))
    def __init__(self, position, fighters:list):
        super(Combat, self).__init__()

        # 一些参数
        self.image = Combat.image
        self.rect = self.image.get_rect()
        self._set_position(position)
        self.s = fighters   # 参与战斗的士兵，有顺序
        self.chaos = 10  # 混战程度，混战程度越大，越难脱离战斗

    def _set_position(self, wolrd_position): # 屏幕移动也会导致屏幕坐标变化，但不会导致世界坐标变化，因此我们这里只设置跟世界坐标有关的部分
        self.world_position = wolrd_position
        self.rect.center = self.world_position

    def update(self, *args: Any, **kwargs: Any) -> None:
        # 混战程度增加
        self.chaos += 1
        if self.chaos > min([s.current_soldier_nums for s in self.s]):
            self.chaos = min([s.current_soldier_nums for s in self.s])

class RemoteCombat(pygame.sprite.Sprite):

    image = pygame.image.load(ARROWS_TARGET)
    def __init__(self, archer, position, attack_range):
        super(RemoteCombat, self).__init__()

        self.image = pygame.transform.scale(RemoteCombat.image, (attack_range, attack_range))
        self.rect = self.image.get_rect()
        self._set_position(position)

        self.archer = archer

    def _set_position(self, wolrd_position): # 屏幕移动也会导致屏幕坐标变化，但不会导致世界坐标变化，因此我们这里只设置跟世界坐标有关的部分
        self.world_position = wolrd_position
        self.rect.center = self.world_position

    def reset_range(self, attack_range):
        self.image = pygame.transform.scale(RemoteCombat.image, (attack_range, attack_range))
        self.rect = self.image.get_rect()
        self._set_position(self.world_position)

    def hit_area(self, spr):    # 打击区域面积
        intersection = spr.rect.clip(self.rect)
        if intersection.width < 0 or intersection.height < 0:
            return 0
        else:
            return intersection.width * intersection.height

    def update(self, soldiers, *args: Any, **kwargs: Any) -> None:

        if not self.archer.remote_clock == self.archer.current_remote_reload_time:
            return

        self.archer.remote_clock = 0

        # 该区域内的所有部队受到平等打击
        for spr in soldiers:
            area = self.hit_area(spr)
            if area != 0:
                spr.current_hurt_by_remote = True
                spr.propose_hurt_by_arrows(self.archer, area)

        self.archer.current_ammo_num -= 1   # 即使是向空地发射，也会浪费弹药
        # 弹药用尽后处理
        if self.archer.current_ammo_num == 0:
            self.archer.set_ammo_exhausted()