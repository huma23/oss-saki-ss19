from traders.dql_trader_helper.state import State
from traders.dql_trader_helper.action import Action

class Unit:
    def __init__(self, state_1:State, action:Action, reward:[], state_2:State):
        self.state_1 = state_1
        self.action = action
        self.reward = reward
        self.state_2 = state_2

class Memory:
    def __init__(self):
        self.list = []

    def add(self, state_1:State, action:Action, reward:[], state_2:State):
        unit = Unit(state_1, action, reward, state_2)
        self.list.append(unit)

    def erase(self):
        self.list = []