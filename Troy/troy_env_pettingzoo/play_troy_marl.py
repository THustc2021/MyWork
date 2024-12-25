import os
import torch
import pygame
from pygame.locals import *
from tianshou.data import Batch
from tianshou.utils.net.common import Net

from train_troy_marl import actor_hidden_sizes, get_env, hidden_dim
from troy_env_pettingzoo.envs.troy_env import TroyEnv
from troy_env_pettingzoo.networks.net import Actor, PreProcessNet
from managers.player_manager import event_handle

MODEL_SAVE_DIR = r"D:\Program-Station\Language-Python\Troy\results\Troy_2agents\DDPGPolicy_DDPGPolicy_pth"
act_random = True
def load_player(enemy_actor_path="", exploration_noise=0):
    env = get_env()
    # 由于observation是一个复合结构，所以我们要对其进行预处理，才能交给preprocess net
    # 我们在preprocess net前再使用一个小型网络，将输入重整并归一化
    observation_space = env.observation_space
    action_space = env.action_space
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    ppprocess_net = PreProcessNet(hidden_dim, observation_space, device=device)
    net = Net(hidden_dim,
              hidden_sizes=actor_hidden_sizes,
              device=device)
    actor = Actor(ppprocess_net, net, action_space.shape or action_space.n,
                  max_actions=action_space.high[0], device=device,
                  exploration_noise=exploration_noise, act_random=act_random).to(device)

    if enemy_actor_path != "":
        st = actor.state_dict()

        load_key = {}
        for k, v in torch.load(enemy_actor_path).items():
            find_key = k.replace("actor.", "")
            if find_key in st.keys():
                load_key[find_key] = v
        print(len(load_key.keys()))
        print(len(st.keys()))

        st.update(load_key)
        actor.load_state_dict(st)

    else:
        actor.act_random = True

    return actor

def start_battle(player_choose, use_AI_enemy=False):

    print("starting battle...")

    if player_choose == 0:
        if use_AI_enemy:
            enemy_actor = load_player(os.path.join(MODEL_SAVE_DIR, "troy_policy_pre.pth"))
        else:
            enemy_actor = load_player()
    else:
        if use_AI_enemy:
            enemy_actor = load_player(os.path.join(MODEL_SAVE_DIR, "greek_policy_pre.pth"))
        else:
            enemy_actor = load_player()

    env = TroyEnv(render_mode="human", event_handler=lambda g, m:event_handle(g, m, g.soldier_manager.troops[player_choose]["troop"],
                                                                              g.soldier_manager.troops[1-player_choose]["troop"]))
    observations, infos = env.reset()

    ename = env.agents[1-player_choose]
    postprocess = lambda x: x[0][0].detach().cpu().numpy()
    while env.agents:
        # this is where you would insert your policy
        actions = {ename: postprocess(enemy_actor(Batch(observations[ename])[None]))}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        # print(rewards)

        if True in terminations.values() or True in truncations.values():
            break

    env.close()
    print("game over.")

if __name__ == '__main__':

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