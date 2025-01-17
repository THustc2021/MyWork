# 基本变量
CHARGE_PATH_NUMS = 96    # 三个tile的距离？
ARROUND_INFLUENCE_DISTANCE = 96 # 周围影响距离
STRENGTH_COMBATING_LOSE = -0.005
STRENGTH_WALKING_LOSE = -0.0001
STRENGTH_RUNNING_LOSE = -0.006
STRENGTH_STANDING_LOSE = 0.0005
STRENGTH_CHARGING_LOSE = -0.007
STRENGTH_DROP_LEVEL1 = 7.5
STRENGTH_DROP_LEVEL2 = 5.0
STRENGTH_DROP_LEVEL3 = 2.5
STRENGTH_DROP_LEVEL4 = 1.0
STRENGTH_DROP_STANDARD = 10
MORALE_DROP_LEVEL1 = 7.5
MORALE_DROP_LEVEL2 = 5.0
MORALE_DROP_LEVEL3 = 3.75
MORALE_DROP_LEVEL4 = 0.1
MORALE_DROP_STANDARD = 10
MAX_RANGE = 512

# 作战影响
PENALTY_MOVING_FROM_BATTLE = 0.95
COMBAT_RESULTING_TIME = 10

# 特殊能力影响
EFFECTIVE_FIGHTER_IN_DENSE = 50
IN_DENSE_ATTACK_MULTI_RATIO = 0.9
IN_DENSE_DEFENSE_MULTI_RATIO = 1.2
EFFECTIVE_FIGHTER_IN_TESTUDO = 25
IN_TESTUDO_ATTACK_MULTI_RATIO = 0.9
IN_TESTUDO_DEFENSE_MULTI_RATIO = 1.5
STRIKE_ATTACK_MULTI_RATIO = 1.5
STRIKE_DEFENSE_MULTI_RATIO = 0.9
EFFECTIVE_COURAGE_RANGE = 256   # 激励范围
EFFECTIVE_THREATEN_RANGE = 128  # 恐吓范围
EFFECTIVE_COMMAND_RANGE = 128   # 指挥范围
COMMAND_MULTIPLY_RATIO = 1.15    # 一般指挥加成
COMMAND_ORDER_MULTIPLT_RATIO = 1.3  # 军事指挥命令加成
REMOTE_PRECISE_RATIO = 0.75

# 时钟控制
SHOW_CLOCK_NORMAL = 1
SHOW_CLOCK_SURRENDING = 2
COLD_CLOCK_ABILITY = 480    # 技能冷却时间
MAINTAIN_CLOCK_ABILITY = 240    # 技能持续时间

# 特殊能力，0代表被动技能，1代表主动技能，2表示特殊技能
IN_DENSE = ("密集阵型", 1)
IN_TESTUDO = ("龟甲阵", 1)
STRIKE_ATTACK = ("突击作战", 1)
COURAGE_ABILITY = ("激励能力", 0)
THREATEN_ENEMY_ABILITY = ("震慑敌军能力", 0)
BROKE_DENSE_ABILITY = ("破坏密集阵型能力", 0)
REMOTE_ATTACK_ABILITY = ("远程攻击能力", 0)
REMOTE_ATTACK_PRECISE = ("精确射击", 1)
COMMAND_ORDER = ("军事指挥", 2)
COMMAND_ABILITY = ("指挥能力", 0)
IMMUNE_THREAT = ("免疫敌军恐吓", 0)
GLORY_ABILITY = ("免疫寡不敌众", 0)