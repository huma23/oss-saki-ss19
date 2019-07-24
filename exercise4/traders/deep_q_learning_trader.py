import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from traders.dql_trader_helper.state import State
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
from traders.dql_trader_helper.action import Action


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 4
        self.action_size = 9
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.2
        self.gamma = 0.1
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        #Compute current State
        state_now = State(stock_market_data, self.expert_a, self.expert_b, self.last_state)
        if (self.last_state == None):
            state_now.prev_state = state_now

        #Create actions and let probability decide
        rand_action = Action.create_random_action()
        model_action = self.action_by_model(state_now)
        action = np.random.choice([rand_action, model_action], 1, p=[self.epsilon, 1-self.epsilon])[0]

        if(self.epsilon > self.epsilon_min):
            self.epsilon -= self.epsilon_decay

        #if training is deactivated
        if not self.train_while_trading:
            self.last_state = state_now
            return model_action.create_order_list(portfolio, stock_market_data)

        #for the first call of trade()
        if(self.last_state == None):
            self.epsilon = 1.0
            self.last_state = state_now
            self.last_portfolio_value = portfolio.get_value(stock_market_data)
            self.last_action = action
            return action.create_order_list(portfolio, stock_market_data)

        #calculate reward and create tuple for memory
        reward = self.create_reward(portfolio, stock_market_data, state_now)
        memory_unit = (self.last_state, self.last_action, reward, state_now)
        self.memory.append(memory_unit)

        #train if there is enough experience
        if len(self.memory) > self.min_size_of_memory_before_training:
            #start training with random batch
            batch = random.sample(self.memory, self.batch_size)
            x = np.empty((self.batch_size, self.state_size))
            y = np.empty((self.batch_size, self.action_size))
            i = 0
            for (s1, a, r, s2) in batch:
                x[i] = s1.create_array_diff()
                y[i] = r
                i += 1

            self.model.train_on_batch(x, y)

        #save old values for next run
        self.last_state = state_now
        self.last_portfolio_value = portfolio.get_value(stock_market_data)
        self.last_action = action
        return action.create_order_list(portfolio, stock_market_data)

    def create_reward(self, portfolio:Portfolio, stock_market_data:StockMarketData, state_now:State):
        new_portfolio_value = portfolio.get_value(stock_market_data)
        index_of_action = Action.get_id_from_action(self.last_action)
        reward = -1 if (self.last_portfolio_value - new_portfolio_value) > 0 else 1
        reward = 0 if (self.last_portfolio_value - new_portfolio_value) == 0 else reward
        reward_array = np.zeros([9])
        reward_array[index_of_action] = reward

        q_next = self.run_model(state_now)
        weighted_q_next = q_next * self.gamma
        reward_array = np.sum([reward_array, weighted_q_next], axis=0)
        #reward_array[index_of_action] += self.gamma * q_next[index_of_action]
        return reward_array

    def run_model(self, input:State):
        state_array = input.create_array_diff()
        out = self.model.predict(np.array([state_array]))
        return out[0]

    def action_by_model(self, input:State):
        q_next = self.run_model(input)
        max = q_next[0]
        index = 0
        for i in range(9):
            if q_next[i] > max:
                max = q_next[i]
                index = i
        #index = np.where(q_next[0] == np.amax(q_next[0]))[0][0]
        action = Action.create_action_from_id(index)
        return action

# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()
