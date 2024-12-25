import os.path

import pygame
import time
from pygame.locals import *

from config_assert import KNIFE_XBM
from config_values import *

''' 用户做的事情：
1、退出游戏
2、移动己方单位（移至某一位置/追逐敌人/脱离战斗）
3、移动镜头
4、鼠标悬停敌方单位改变样式
'''

knife_cursor = pygame.cursors.load_xbm(KNIFE_XBM, KNIFE_XBM)
def event_handle(gameGroup, gameMap, player_troop, enemy_troop):

    for event in pygame.event.get():
        # 鼠标悬停事件
        mouse_pos = pygame.mouse.get_pos()
        mouse_world_pos = gameGroup.screen_to_world(*mouse_pos)
        hover_enemy = None
        for e in enemy_troop:
            if e.rect.collidepoint(mouse_world_pos):
                hover_enemy = e # 找最上层的敌人，不额外遍历
                break
        if hover_enemy is None:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)
        else:
            pygame.mouse.set_cursor(*knife_cursor)

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
            if event.button == 1:  # 左键点击
                # 遍历组中的精灵，判断是否有精灵被点击，点击位置离谁的中心最近选的就是谁
                min_dist = -1
                choosed_spr = None
                for sprite in player_troop:
                    if sprite.rect.collidepoint(mouse_world_pos) and not sprite.is_surrending():  # 通过此法可以定位精灵
                        # 计算距离
                        c = sprite.rect.center
                        dist = (mouse_world_pos[0] - c[0]) ** 2 + (mouse_world_pos[1] - c[1]) ** 2
                        if min_dist == -1 or dist < min_dist:
                            choosed_spr = sprite
                # 设置选中
                if choosed_spr != None:
                    choose_flag = True
                    gameGroup.set_choose(choosed_spr)
                elif gccs != None and gccs.ability_enabled(REMOTE_ATTACK_ABILITY) \
                        and gccs.check_in_remote_range(mouse_world_pos):
                    # 左键点击以进行区域射击
                    gameGroup.add_remote_combat(gccs, mouse_world_pos, gccs.current_remote_attack_range())
                    choose_flag = True
            elif event.button == 3:  # 右键点击
                # 移动
                if gccs != None:
                    if hover_enemy != None:
                        gccs.command_chasing_enemy(hover_enemy)
                    else:
                        world_position = gameGroup.screen_to_world(*mouse_pos)
                        gccs.command_moving_position(world_position)
                choose_flag = True  # 保持选中
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