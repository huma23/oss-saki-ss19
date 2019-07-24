import numpy as np
from framework.portfolio import Portfolio, StockMarketData
from framework.company import Company
from framework.interface_expert import Vote, IExpert

class State:
    def __init__(self, stock_market_data:StockMarketData, expert_a:IExpert, expert_b:IExpert, prev_state):
        self.prev_state = prev_state
        self.price_a = stock_market_data.get_most_recent_price(Company.A)
        self.price_b = stock_market_data.get_most_recent_price(Company.B)
        stock_data_a = stock_market_data[Company.A]
        self.vote_a = expert_a.vote(stock_data_a)
        stock_data_b = stock_market_data[Company.B]
        self.vote_b = expert_b.vote(stock_data_b)

    def create_array(self) -> np.ndarray:
        vote_a_int = self.convert_vote_to_int(self.vote_a)
        vote_b_int = self.convert_vote_to_int(self.vote_b)
        return np.array([self.price_a, self.price_b, vote_a_int, vote_b_int])

    def create_array_diff(self) -> np.ndarray:
        price_diff_a = self.prev_state.price_a - self.price_a
        price_diff_b = self.prev_state.price_b - self.price_b
        vote_a_int = self.convert_vote_to_int(self.vote_a)
        vote_b_int = self.convert_vote_to_int(self.vote_b)
        return np.array([price_diff_a, price_diff_b, vote_a_int, vote_b_int])

    @staticmethod
    def convert_vote_to_int(vote:Vote) -> int:
        if(vote == Vote.HOLD):
            return 0
        if(vote == Vote.BUY):
            return 1
        if(vote == Vote.SELL):
            return -1
        return 0