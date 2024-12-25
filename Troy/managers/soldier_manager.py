from config_values import *
from utils.common import sigmoid


class SoldierManager():

    def __init__(self):

        self.troops = [
            {"troop": [], "king": None, "hero": None, "courage_group": [], "threaten_group": [], "command_group": [], "kdclock": 0, "hdclock":0 , "kdratio": 1, "hdratio": 1},
            {"troop": [], "king": None, "hero": None, "courage_group": [], "threaten_group": [], "command_group": [], "kdclock": 0, "hdclock":0 , "kdratio": 1, "hdratio": 1}
        ]

    def add_soldier(self, soldier, which_troop, is_king=False, is_hero=False):
        self.troops[which_troop]["troop"].append(soldier)
        if is_king:
            self.troops[which_troop]["king"] = soldier
        if is_hero:
            self.troops[which_troop]["hero"] = soldier

    def set_all_kh_ratio(self, clock, r=200):
        for troop in self.troops:
            if troop["king"] == None:
                troop["kdratio"] = sigmoid(1 - (clock - troop["kdclock"]) / r)  + 1
            if troop["hero"] == None:
                troop["hdratio"] = sigmoid(1 - (clock - troop["hdclock"]) / r)  + 1
    def get_his_kh_ratio(self, troop_id, king_or_hero):
        return self.troops[troop_id]["kdratio" if king_or_hero == 0 else "hdratio"]

    def remove_soldier(self, soldier, clock=None):
        troop = self.troops[soldier.troop_id]
        if troop["king"] == soldier:
            troop["king"] = None
            troop["kdclock"] = clock
        if troop["hero"] == soldier:
            troop["hero"] = None
            troop["hdclock"] = clock
        troop["troop"].remove(soldier)

    def get_his_king(self, tid):
        return self.troops[tid]["king"]
    def get_his_hero(self, tid):
        return self.troops[tid]["hero"]

    def get_his_enemy_troop(self, troop_id):    # 在多个troop的情况下，应当额外设置
        return self.troops[1-troop_id]["troop"]
    def get_his_troop(self, troo_id):
        return self.troops[troo_id]["troop"]

    def check_any_troop_destroyed(self):
        for troop_id in range(len(self.troops)):
            flag = True
            for ps in self.troops[troop_id]["troop"]:
                if not ps.is_surrending():
                    flag = False
                    break
            if flag:
                return troop_id
        return -1
