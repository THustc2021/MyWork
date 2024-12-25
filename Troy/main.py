import time

import pygame
from pygame.locals import *

from managers.map_manager import MainMap
from managers.player_manager import event_handle
from managers.enemy_manager import enemy_movement
from managers.display_manager import MainGroup
from units import troy_soldiers, greece_soldiers

def start_battle(player_choose, win_width=1080, win_height=640):

    print("starting battle...")
    gameMap = MainMap('assert/maps/grasslands/grasslands.tmx', (win_width, win_height))
    gameGroup = MainGroup(gameMap.map_layer, (win_width, win_height))

    troy_soldiers.Infrantry((300, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Infrantry((360, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Infrantry((420, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Infrantry((480, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Infrantry((540, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Infrantry((600, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Infrantry((660, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Infrantry((720, 400), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Rider((800, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Rider((860, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Rider((920, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Rider((980, 600), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Archer((160, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Archer((220, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Archer((280, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Archer((340, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Archer((400, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Archer((460, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Archer((520, 300), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.TroyGuard((260, 500), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.TroyGuard((400, 500), gameGroup=gameGroup, gameMap=gameMap, which_troop=0)
    troy_soldiers.Hector((340, 500), gameGroup=gameGroup, gameMap=gameMap, which_troop=0, is_hero=True)
    troy_soldiers.Priam((150, 200), gameGroup=gameGroup, gameMap=gameMap, which_troop=0, is_king=True)

    greece_soldiers.GreeceArcher((800, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.GreeceArcher((860, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.GreeceArcher((920, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.GreeceArcher((980, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.AthensMan((1040, 1200), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.AthensMan((1100, 1200), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.MopoliaArcher((1200, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.MopoliaArcher((1260, 1200), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.MycenaeanSworder((800, 1300), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.MycenaeanSworder((860, 1300), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.MycenaeanSworder((920, 1300), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.MycenaeanSworder((980, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.MycenaeanSworder((1040, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.MycenaeanSworder((1100, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.MycenaeanSworder((1160, 1300), gameGroup=gameGroup, gameMap=gameMap,
                                     which_troop=1)
    greece_soldiers.ArgosMan((860, 1400), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.ArgosMan((920, 1400), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.PylosMan((720, 1300), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.PylosMan((1240, 1300), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.SpartarMan((980, 1400), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.SpartarMan((1040, 1400), gameGroup=gameGroup, gameMap=gameMap, which_troop=1)
    greece_soldiers.Achilles((1000, 1500), gameGroup=gameGroup, gameMap=gameMap, which_troop=1, is_hero=True)
    greece_soldiers.AkaMenon((1080, 1500), gameGroup=gameGroup, gameMap=gameMap, which_troop=1, is_king=True)

    # set player and enemy
    player_troop = gameGroup.soldier_manager.troops[player_choose]["troop"]
    enemy_troop = gameGroup.soldier_manager.troops[1-player_choose]["troop"]
    enemy_king = gameGroup.soldier_manager.troops[1 - player_choose]["king"]

    ### main loop
    screen = pygame.display.set_mode((win_width, win_height))

    player_won = None
    while player_won is None and not gameGroup.game_exit:
        # 处理移动和展示
        b = time.time()
        gameGroup.update()  # update的时间是整个游戏的瓶颈
        gameGroup.draw(screen)
        pygame.display.flip()
        e = time.time()
        sleep_time = 1/20-(e-b)
        time.sleep(sleep_time if sleep_time > 0 else 0)
        # 更新时钟，
        if not event_handle(gameGroup, gameMap, player_troop, enemy_troop):
            gameGroup.game_exit = True
            break
        enemy_movement(enemy_troop, enemy_king, player_troop)
        # 胜利目标
        did = gameGroup.soldier_manager.check_any_troop_destroyed()
        if not did == -1:
            if did == player_choose:
                player_won = False
            else:
                player_won = True

    # 结算
    if player_won != None:
        if player_won:
            print("player won")
        else:
            print("player lose")
    print("game over.")

def main():

    win_width, win_height = 550, 420
    import os
    os.environ['SDL_VIDEODRIVER'] = 'windows'
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 200)

    pygame.init()
    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption('Troy')
    pygame.display.set_icon(pygame.image.load("assert/icon.png"))
    clock = pygame.time.Clock()

    # 背景
    background = pygame.transform.scale(pygame.image.load("assert/background.png").convert(), (win_width, win_height))
    screen.blit(background, (0, 0))

    # 字体
    font = pygame.font.SysFont('simHei', 36)
    choose_country = font.render("选择势力", True, (255, 255, 255))
    screen.blit(choose_country, (200, 50))

    # 选择势力
    font2 = pygame.font.SysFont('simHei', 20)
    gbutton = pygame.surface.Surface((200, 205))
    gbutton.fill((255, 255, 255))
    gimage = pygame.transform.scale(pygame.image.load("assert/greek_king.png").convert(), (200, 180))
    greek_ = font2.render("希腊联军", True, (96, 96, 96))
    gbutton.blit(gimage, (0, 0))
    gbutton.blit(greek_, (60, 180))

    tbutton = pygame.surface.Surface((200, 205))
    tbutton.fill((255, 255, 255))
    timage = pygame.transform.scale(pygame.image.load("assert/troy_king.png").convert(), (200, 180))
    troy_ = font2.render("特洛伊", True, (0, 0, 255))
    tbutton.blit(timage, (0, 0))
    tbutton.blit(troy_, (72, 180))

    grect = screen.blit(gbutton, (50, 140))
    trect = screen.blit(tbutton, (300, 140))

    pygame.display.flip()

    running = True
    player_choose = None
    while running and player_choose == None:  # 静态界面
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键按下
                    if grect.collidepoint(event.pos):
                        player_choose = 1
                        print("player choose greek!")
                        break
                    elif trect.collidepoint(event.pos):
                        player_choose = 0
                        print("player choose troy!")
                        break
        # 获取鼠标的位置
        mouse_pos = pygame.mouse.get_pos()
        if grect.collidepoint(mouse_pos) or trect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(*pygame.cursors.broken_x)
        else:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)

        clock.tick(20)

    if player_choose != None:
        # 清空屏幕
        screen.fill((0, 0, 0))
        pygame.mouse.set_cursor(*pygame.cursors.arrow)
        # 展示选择
        if player_choose == 1:
            t = font.render("特洛伊", True, (0, 0, 255))
            screen.blit(t, (180, 200))
        else:
            t = font.render("希腊联军", True, (96, 96, 96))
            screen.blit(t, (220, 200))
        tt = font2.render("正在载入...", True, (255, 255, 255))
        screen.blit(tt, (180, 380))
        pygame.display.flip()
        # 开始游戏
        start_battle(player_choose)

    pygame.quit()

if __name__ == '__main__':

    main()