import random
import time

from config_values import *
# 战斗逻辑
def enemy_movement(enemy_troop, enemy_king, player_troop):

    # 这里我们先让他随机移动
    for e in enemy_troop:
        if e == enemy_king:
            # 跟随移动
            e.command_moving_position(random.choice(enemy_troop).world_position)
            continue
        if not e.in_close_combat() and len(e.moving_position) == 0 and e.chasing_target==None:
            e.command_chasing_enemy(random.choice(player_troop))
        else:
            # 随机发动技能
            if random.random() > 0.5:
                e.command_ability1()
            else:
                e.command_ability2()