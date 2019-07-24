import itertools
import random

from framework.portfolio import Portfolio
from framework.order import OrderType, Order
from framework.company import Company
from framework.stock_market_data import StockMarketData
from framework.interface_expert import Vote

class Action:
    def __init__(self, type_a:Vote, type_b:Vote):
        self.type_a = type_a
        self.type_b = type_b

    def create_order_list(self, portfolio:Portfolio, stock_market_data:StockMarketData) -> [Order]:
        list = []
        order_a = self.__create_order_for_company(Company.A, portfolio, self.type_a, stock_market_data)
        if order_a != None:
            list.append(order_a)
        order_b = self.__create_order_for_company(Company.B, portfolio, self.type_b, stock_market_data)
        if order_b != None:
            list.append(order_b)
        return list

    def __create_order_for_company(self, company:Company, portfolio:Portfolio, vote:Vote, stock_market_data:StockMarketData) -> Order:
        order = None
        if vote == Vote.SELL:
            amount = portfolio.get_stock(company)
            if amount > 0:
                order = Order(OrderType.SELL, company, amount)
        elif vote == Vote.BUY:
            stock_price = stock_market_data.get_most_recent_price(company)
            amount = 0
            if (self.type_a == self.type_b):
                # buy both - half portfolio value for each
                amount = int((portfolio.cash // 2) // stock_price)
            else:
                amount = int(portfolio.cash // stock_price)

            if amount > 0:
                order = Order(OrderType.BUY, company, amount)
        return order

    @staticmethod
    def action_type_dictionary() -> {int: (Vote, Company)}:
        list_a = list(itertools.product(Vote, [Company.A]))
        list_b = list(itertools.product(Vote, [Company.B]))
        combinations = list(itertools.product(list_a, list_b))

        dict = {}
        i = 0
        for comb in combinations:
            dict[i] = comb
            i += 1
        return dict

    @staticmethod
    def get_action_type_from_index(index: int) -> (Vote, Company):
        return Action.action_type_dictionary()[index]

    @staticmethod
    def get_random_action_type() -> (Vote, Company):
        return Action.get_action_type_from_index(random.randint(0, 8))

    @staticmethod
    def create_random_action():
        random_action_type = Action.get_random_action_type()
        action = Action(random_action_type[0][0], random_action_type[1][0])
        return action

    @staticmethod
    def create_action_from_id(index:int):
        action_type = Action.get_action_type_from_index(index)
        action = Action(action_type[0][0], action_type[1][0])
        return action

    @staticmethod
    def get_id_from_action(action) -> int:
        dict = Action.action_type_dictionary()
        for i in range(9):
            action_type = dict[i]
            action_type_a = action_type[0]
            action_type_b = action_type[1]
            if action_type_a[0] == action.type_a:
                if action_type_b[0] == action.type_b:
                    return i



