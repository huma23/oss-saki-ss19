import copy
import numpy as np
import datetime
import itertools
import mdptoolbox
import gc
import pandas as pd
from enum import IntEnum, Enum

class Color(IntEnum):
    WHITE = 0
    BLUE = 1
    RED = 2

class OccupancyType(IntEnum):
    WHITE = 0
    BLUE = 1
    RED = 2
    EMPTY = 3

class InputAction(Enum):
    STORE = 0
    RESTORE = 1

class Input:
    def __init__(self, input_action:InputAction, color:Color):
        self.input_action = input_action
        self.color = color

    def get_id(self):
        return len(Color) * self.input_action.value + self.color.value

    def print(self):
        print("inputaction: {} - color: {}".format(self.input_action, self.color))

    @staticmethod
    def combinations():
        return list(itertools.product(InputAction, Color))

class Action:
    def __init__(self, pos_x:int, pos_y:int):
        self.pos_x = pos_x
        self.pos_y = pos_y

class Occupancy:
    def __init__(self, height:int, length: int):
        self.state = [OccupancyType.EMPTY] * (height * length)
        self.length = length

    def perform_action(self, action:Action, input:Input):
        new_occ = copy.deepcopy(self)
        index = (action.pos_x-1) * new_occ.length + (action.pos_y - 1)
        error = False

        if input.input_action == InputAction.STORE:
            if (new_occ.state[index] != OccupancyType.EMPTY):
                error = True
            new_occ.state[index] = OccupancyType(input.color)
        else:
            if new_occ.state[index] != input.color:
                error = True
            new_occ.state[index] = OccupancyType.EMPTY
        return new_occ, error

    def combinations(self):
        return list(itertools.product(OccupancyType, repeat=len(self.state)))

    def get_id(self):
        id = 0
        for index in range(0, len(self.state)):
            id += np.power(len(OccupancyType), index) * self.state[index]
        return id;

    def print(self):
        print("[")
        for x in self.state:
            print(x)
        print("]")

class WareHouse:
    def __init__(self, rows:int, columns: int):
        self.rows = rows
        self.columns = columns
        self.inputs = self.create_all_inputs()
        self.actions = self.create_all_actions()
        self.states = self.create_only_logical_states()

    def get_occupancy_length(self):
        return self.rows * self.columns

    def create_all_inputs(self):
        inputs = []
        for ia in InputAction:
            for c in Color:
                new_input = Input(ia, c)
                inputs.append(new_input)
        return inputs

    def create_all_actions(self):
        actions = []
        for r in range(1, self.rows + 1):
            for c in range(1, self.columns + 1):
                actions.append(Action(r,c))
        return actions

    def create_only_logical_states(self):
        logical_states = []
        states = self.create_all_states()
        self.state_row_dictionary = {}
        rowIndex = 0
        for state in states:
            if state.is_state_logical():
                logical_states.append(state)
                self.state_row_dictionary[state.get_id()] = rowIndex
                rowIndex += 1
        return logical_states

    def create_all_states(self):
        self.state_row_dictionary = {}
        list = []
        inputs = Input.combinations()
        sample = self.create_empty_occupancy()
        occs = sample.combinations()
        rowIndex = 0
        for occ in occs:
            for input in inputs:
                occupancy = self.create_defined_occupancy(np.array(occ))
                inp = Input(input[0], input[1])
                s = State(self, occupancy, inp)
                list.append(s)
                self.state_row_dictionary[s.get_id()] = rowIndex
                rowIndex += 1
        return list

    def create_empty_occupancy(self):
        occ = Occupancy(self.rows, self.columns)
        return occ

    def create_defined_occupancy(self, state:[OccupancyType]):
        occ = self.create_empty_occupancy()
        occ.state = state
        return occ

    def get_possible_occupancy_count(self):
        return np.power(len(OccupancyType), self.get_occupancy_length())

    def get_states_count(self):
        return len(self.states)

class OrderFile:

    def __init__(self, filename:str):
        length = len(Input.combinations())
        self.matrix = np.zeros((length, length))
        self.count = {}
        self.count_sum = 0
        self.duration = {}
        self.inputs = self.read_file(filename)
        return

    def get_frequency_of_color(self, color:Color):
        return self.count[color] / self.count_sum

    #returns the percentage how much the given color has a longer duration than the minimum duration color
    def get_duration_ratio(self, color:Color):
        lowest = min(self.duration.values())
        ratio = (self.duration[color] / lowest) - 1
        return ratio

    def create_statistic(self, inputs:[Input]):
        duration_count = {
            Color.WHITE: 0,
            Color.BLUE: 0,
            Color.RED: 0
        }
        duration_lists = {
            Color.WHITE : [],
            Color.BLUE : [],
            Color.RED : []
        }
        storage_count = {
            Color.WHITE: 0,
            Color.BLUE: 0,
            Color.RED: 0
        }
        prev_input:Input = None
        for input in inputs:

            if prev_input != None:
                self.matrix[prev_input.get_id(), input.get_id()] += 1
            prev_input = input

            if(input.input_action == InputAction.STORE):
                if input.color not in self.count:
                    self.count[input.color] = 0
                self.count[input.color] += 1
                self.count_sum += 1
                storage_count[input.color] += 1

            else:
                storage_count[input.color] -= 1
                if storage_count[input.color] == 0:
                    duration_lists[input.color].append(duration_count[input.color])
                    duration_count[input.color] = 0

            for key in storage_count:
                if storage_count[key] != 0:
                    duration_count[key] += 1

        for key in duration_lists:
            self.duration[key] = np.mean(duration_lists[key])

    def read_file(self, filename:str):
        inputs = []
        with open(filename, 'r') as file:
            for line in file:
                split = line.rstrip().split("\t")
                inputAction = InputAction[split[0].upper()]
                color = Color[split[1].upper()]
                inputs.append(Input(inputAction, color))
        self.create_statistic(inputs)
        return inputs

class State:
    def __init__(self, warehouse:WareHouse, occupancy:Occupancy, input:Input):
        self.warehouse = warehouse
        self.occupancy = occupancy
        self.input = input

    def get_next_state(self, action:Action, new_input:Input):
        #state_copy = copy.deepcopy(self)
        state_copy = State(self.warehouse, self.occupancy, new_input)
        new_occupancy, error = state_copy.occupancy.perform_action(action, self.input)
        state_copy.occupancy = new_occupancy
        if not error:
            if not state_copy.is_state_logical():
                error = True
        return state_copy, error

    def get_next_states(self, action:Action):
        new_states = []
        errors = []
        for new_input in self.warehouse.inputs:
            new_state, error = self.get_next_state(action, new_input)
            new_states.append(new_state)
            errors.append(error)
        return new_states, errors

    def get_next_possible_states(self, action:Action):
        new_states = []
        new_errors = []
        all_states, all_errors = self.get_next_states(action)
        for i in range(0, len(all_errors)):
            if(all_errors[i] != True):
                new_states.append(all_states[i])
                new_errors.append(all_errors[i])
        return new_states

    def get_all_next_states(self):
        new_states = []
        new_errors = []
        for action in self.warehouse.actions:
            states, errors = self.get_next_states(action)
            for i in range(0, len(errors)):
                new_states.append(states[i])
                new_errors.append(errors[i])
        return new_states, new_errors

    def get_id(self):
        id_occ = self.occupancy.get_id()
        id_input = self.input.get_id()
        return id_occ + self.warehouse.get_possible_occupancy_count() * id_input

    def print(self):
        self.occupancy.print()
        self.input.print()

    def is_state_logical(self):
        hasEmpty = False
        completeEmpty = True
        for occ in self.occupancy.state:
            if occ == OccupancyType.EMPTY:
                hasEmpty = True
            else:
                completeEmpty = False

        # all impossible scenarios
        if (self.input.input_action == InputAction.STORE) and (hasEmpty == False):
            return False

        if self.input.input_action == InputAction.RESTORE:
            if (completeEmpty == True):
                return False
            if not (self.input.color in self.occupancy.state):
                return False
        return True

class Probabilities:
    @staticmethod
    def get_probability_equal(current_state:State, possible_states:[State], order_file:OrderFile):
        #first of all let's try equal probabilities for each userinput
        list = []
        prob = round(1.0/len(possible_states), 4)
        for state in possible_states:
            t = (state, prob)
            list.append(t)
        return list

    @staticmethod
    def get_probability_by_distribution(current_state:State, possible_states:[State], order_file:OrderFile):
        if orderFile == None:
            return Probabilities.get_probability_equal(current_state, possible_states, None)

        values = []
        sum = 0
        for state in possible_states:
            val = order_file.matrix[current_state.input.get_id(), state.input.get_id()]
            values.append((state, val))
            sum += val
        tuples = []
        for (state, value) in values:
            prob = value / sum
            prob = round(prob, 4)
            t = (state, prob)
            tuples.append(t)

        return tuples

class Rewards:
    def __init__(self, states_count:int, actions:[Action], order_file:OrderFile):
        self.matrix = np.zeros((states_count, len(actions)))
        self.order_file = order_file
        self.actions = actions

    def get_reward(self, current_state:State, action:Action, frequency:bool=True, duration:bool=True):
        possible_states = current_state.get_next_possible_states(action)

        if len(possible_states) == 0:
            return -10

        #simple evaluation of the distance the action would bring
        distance_value = self.get_distance_action_reward(current_state, action) * 10

        #combine with frequency & duration of color
        frequency = self.order_file.get_frequency_of_color(current_state.input.color) * 3
        duration = self.order_file.get_duration_ratio(current_state.input.color)

        #distance value is the most important influence on the reward
        #the higher the frequency of a color the higher should be the overall value
        #the longer the duration of the color the lower should be the overall value
        overall_value = distance_value \

        if frequency:
            overall_value += frequency * distance_value \

        if duration:
            overall_value -= duration * distance_value

        return round(overall_value)

    #returns between 0 and height*length-1, the more the better
    def get_distance_action_reward(self, current_state:State, action:Action):
        x = action.pos_x
        y = action.pos_y
        calculated_factor = x*y
        max = current_state.warehouse.rows * current_state.warehouse.columns
        return (max - calculated_factor)

    def calculate_matrix(self, list:[State], state_row_dict, frequency:bool=True, duration:bool=True):
        for state in list:
            i = 0
            for action in self.actions:
                reward = self.get_reward(state, action)
                row_index = state_row_dict[state.get_id()]
                self.matrix[row_index, i] = reward
                i += 1
        return self.matrix

class TransitionMatrix:
    def __init__(self, states_count:int, action:Action):
        #initialize transition matrix with zeros first
        self.matrix = np.zeros((states_count, states_count))
        self.action = action

    def calculate_matrix(self, list:[State], state_row_dict, probability_function, order_file:OrderFile):
        for state in list:
            id = state.get_id()
            row_index = state_row_dict[id]
            next_states = state.get_next_possible_states(self.action)
            if not next_states:
                #no further step with this state so probability is 1 for staying in this state
                self.matrix[row_index, row_index] = 1
            else:
                # set probabilities for this specific action in this state
                probabilitíes = probability_function(state, next_states, order_file)
                for (state2, prob) in probabilitíes:
                    id2 = state2.get_id()
                    if id2 in state_row_dict.keys():
                        row_index2 = state_row_dict[id2]
                        self.matrix[row_index, row_index2] = prob

    def check_fullfillment(self):
        x = 0
        y = 0
        for row in self.matrix:
            sum = np.sum(row)
            if(sum != 1):
                #fix for floating issues
                delta = 1 - sum
                if(delta > 0.1):
                    return False
                else:
                    y = 0
                    for column in row:
                        if column > 0:
                            self.matrix[x, y] += delta
                            break
                        y += 1
            x += 1
        return True


class Evaluation:
    def __init__(self, warehouse: WareHouse, train_file: OrderFile, policy: tuple):
        self.inputs = train_file.inputs
        self.warehouse = warehouse
        self.policy = policy

    def calculate_costs(self):
        empty = warehouse.create_empty_occupancy()
        state = State(warehouse, empty, self.inputs[0])
        costs = 0

        for input in self.inputs[1:]:
            # state.print()
            state_row_index = warehouse.state_row_dictionary[state.get_id()]
            action_id = self.policy[state_row_index]
            action = self.warehouse.actions[action_id]
            state, _ = state.get_next_state(action, input)
            # calc costs - way to the spot and back again
            costs += self.cost(action)

        state_row_index = warehouse.state_row_dictionary[state.get_id()]
        action_id = self.policy[state_row_index]
        action = self.warehouse.actions[action_id]
        costs += self.cost(action)
        mean = costs / len(self.inputs)
        return costs, mean

    def calculate_greedy(self):
        empty = warehouse.create_empty_occupancy()
        state = State(warehouse, empty, self.inputs[0])
        costs = 0

        for input in self.inputs[1:]:
            # state.print()
            for action in warehouse.actions:
                new_state, error = state.get_next_state(action, input)
                if not error:
                    state = new_state
                    costs += self.cost(action)
                    break

        for action in warehouse.actions:
            _, error = state.get_next_state(action, self.inputs[0])
            if not error:
                costs += self.cost(action)
                break

        mean = costs / len(self.inputs)
        return costs, mean

    def cost(self, action: Action):
        return (action.pos_y - 1 + action.pos_x) * 2
        # return action.pos_y * action.pos_x * 2


#main code
variations = list(itertools.product([True, False], repeat=3))
print("=== RESULTS ===")

for (duration, frequency, prob) in variations:
    orderFile = OrderFile("warehousetraining.txt")
    testFile = OrderFile("warehouseorder.txt")
    warehouse = WareHouse(2, 3)

    #calculate reward matrix#
    #print(len(warehouse.states))
    #print("Create Reward")
    #print("Start: ", datetime.datetime.now())
    reward = Rewards(warehouse.get_states_count(), warehouse.actions, orderFile)
    rmatrix = reward.calculate_matrix(warehouse.states, warehouse.state_row_dictionary, frequency=frequency, duration=duration)
    rmatrix = rmatrix.astype(np.int8)
    #print("End: ", datetime.datetime.now())

    #calculate transition for each action
    #print("Create Transitions")
    #print("Start: ", datetime.datetime.now())
    trans = np.zeros((len(warehouse.actions), len(warehouse.states), len(warehouse.states)), dtype=np.float16)
    i = 0
    for action in warehouse.actions:
        matrix = TransitionMatrix(warehouse.get_states_count(), action)
        if(prob):
            matrix.calculate_matrix(warehouse.states, warehouse.state_row_dictionary, Probabilities.get_probability_by_distribution, orderFile)
        else:
            matrix.calculate_matrix(warehouse.states, warehouse.state_row_dictionary, Probabilities.get_probability_equal, orderFile)

        matrix.check_fullfillment()
        trans[i] = matrix.matrix.astype(np.float16)
        del(matrix)
        i += 1
    #print("End: ", datetime.datetime.now())

    #create and train value iteration
    #print("Train")
    #print("Start: ", datetime.datetime.now())
    policies = []
    names = []
    for x in range(3):
        if x == 0:
            names.append("ValueIteration")
            alg = mdptoolbox.mdp.ValueIteration(trans, rmatrix, 0.3, max_iter=100)
        if x == 1:
            names.append("RelativeValueIteration")
            alg = mdptoolbox.mdp.RelativeValueIteration(trans, rmatrix, max_iter=100)
        if x == 2:
            names.append("ValueIterationGS")
            alg = mdptoolbox.mdp.ValueIterationGS(trans, rmatrix, 0.3)
        alg.run()
        policies.append(alg.policy)
        del(alg)

    prob_name = "Distribution of the training set" if prob else "Equally Distributed"
    print("Transition matrix: ", prob_name)
    print("Frequency of color in rewards: ", frequency)
    print("Duration of color in rewards: ", duration)
    costs_greedy = 0
    mean_greedy = 0
    for x in range(3):
        print("Algorithm: ", names[x])
        eval = Evaluation(warehouse, testFile, policies[x])
        costs_value, mean_value = eval.calculate_costs()
        costs_greedy, mean_greedy = eval.calculate_greedy()
        del(eval)
        print('Sum = ', costs_value, ' - Mean = ', mean_value)
    print("Algorithm: Greedy")
    print('Sum = ', costs_greedy, ' - Mean = ', mean_greedy)
    print("===========================")
    print("  ")

    del(policies)
    del(names)
    del(trans)
    del(rmatrix)
    del(reward)
    del(warehouse)
    del(testFile)
    del(orderFile)
    gc.collect()

#mdpresultPolicy = mdptoolbox.mdp.PolicyIteration(trans, rmatrix, 0.3, max_iter=100)
#mdpresultValue = mdptoolbox.mdp.ValueIteration(trans, rmatrix, 0.3, max_iter=100)

#print("train policyiteration:")
#print(datetime.datetime.now())
#mdpresultPolicy.run()
#print(datetime.datetime.now())
#print("train valueiteration:")
#print(datetime.datetime.now())
#mdpresultValue.run()
#print(datetime.datetime.now())

#print("===RESULTS===")
#print('PolicyIteration:')
#print(mdpresultPolicy.policy)
#print(mdpresultPolicy.V)
#print(mdpresultPolicy.iter)

#print('ValueIteration:')
#print(mdpresultValue.policy)
#print(mdpresultValue.V)
#print(mdpresultValue.iter)
#policy = mdpresultValue.policy

#print('Evaluation:')
#testFile = OrderFile("warehouseorder.txt")
#eval = Evaluation(warehouse, testFile, policy)
#costs_greedy, mean_gready = eval.calculate_greedy()
#costs_value1, mean_value1 = eval.calculate_costs()

#print("valueiteration: ", costs_value1, " - ", mean_value1)
#print("greedy: ", costs_greedy, " - ", mean_gready)
