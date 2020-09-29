import numpy as np
from random import randint
import random
from itertools import permutations
from collections import deque
from keras import Sequential
from keras.activations import relu, linear, sigmoid
from keras.layers import Dense
from keras.optimizers import adam
from keras.optimizers import SGD
import copy
import traceback

class Agent:
    # Agent properties
    def __init__(self, action_space, state_space, batch_size, state_space_size):
        self.action_space = action_space  # move to other locations or stay
        self.state_space = state_space    # trailer location, day, trailer size, load vector
        self.epsilon = 1.0  # initial epsilon, probability of taking random action
        self.gamma = .99  # discount factor of bellman equation
        self.batch_size = batch_size
        self.epsilon_min = .1 # min epsilon values
        self.lr = 0.01  # learning rate of the agent
        self.epsilon_decay = .995
        self.state_space_size = state_space_size
        self.memory = deque(maxlen=20000)  # replay memory
        self.model = self.build_model()
        #self.align_model()
        self.update_freq = 4

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_space, activation=relu))
        model.add(Dense(128, activation=relu))
        model.add(Dense(128, activation=relu))
        model.add(Dense(64, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        opt = SGD(lr=self.lr, momentum=0.9, clipnorm=1.0)
        model.compile(loss='mse', optimizer=opt)
        return model

    # Neural Network Architecture for test
    def build_test_model(self, state_space_size, batch_size):
        test_model = Sequential()
        test_model.add(Dense(64, input_dim=self.state_space, activation=relu))
        test_model.add(Dense(128, activation=relu))
        test_model.add(Dense(128, activation=relu))
        test_model.add(Dense(64, activation=relu))
        test_model.add(Dense(self.action_space, activation=linear))
        test_model.load_weights(str(batch_size)+'weights'+str(state_space_size)+'.h5')
        opt = SGD(lr=self.lr, momentum=0.9, clipnorm=1.0)
        test_model.compile(loss='mse', optimizer=opt)
        return test_model

    def align_model(self):
        self.model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )

    def act (self, state, action_space):
        trailer_location, day_of_week, trailer_size, load_vector = state
        # explore case
        if np.random.rand() <= self.epsilon:
            random_action_index = np.random.choice(action_space)

            return random_action_index

        # exploit case
        # flatten the state values in order to use as a neural network input

        state_flatten = []

        for i in range(3):
            state_flatten.append(state[i])

        for i in range(action_space):
            state_flatten.append(state[3][i])

        # make prediction and return the action which has maximum value
        Q_values = self.model.predict(np.array(state_flatten).reshape(1, self.state_space))
        best_action_index=np.argmax(Q_values[0])

        return best_action_index

    def replay(self, action_space):

        if len(self.memory) < self.batch_size:

            return

        # take a minibatch from experiences

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])

        # flatten states:
        state_list = []

        for i in states:
            temp_list = []
            for j in range(3):
                temp_list.append(i[j])
            for k in range(action_space):
                temp_list.append(i[3][k])
            state_list.append(temp_list)

        states = np.array(state_list)
        actions = np.array([i[1] for i in minibatch])

        # rewards after each action
        rewards = np.array([i[2] for i in minibatch])

        # check episode is done or not
        dones = np.array([i[4] for i in minibatch])

        # calculate next states
        next_states = np.array([i[3] for i in minibatch])

        # flatten next states
        next_state_list = []

        for i in next_states:
            next_temp_list = []
            for j in range(3):
                next_temp_list.append(i[j])
            for k in range(action_space):
                next_temp_list.append(i[3][k])
            next_state_list.append(next_temp_list)
        next_states = np.array(next_state_list)

        # flatten states and next states
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)

        # predicted q values for current batch with current network
        targets_full = self.model.predict_on_batch(states)

        # indices for each element in batch
        ind = np.array([i for i in range(self.batch_size)])

        # update corresponding action state pairs with expected q values
        targets_full[[ind], [actions]] = targets

        # train the network with states and updated q values corresponding to actions
        result = self.model.fit(states, targets_full, epochs=1, verbose=0)
        return result.history['loss'][0]

# same nomadic trucker model used previously in value iteration, qlearning and benchmark policy
class Environment:
    def __init__(self, agent, size, num_days, num_trailers):
        self.agent = Agent
        self.days_of_week = [1, 2, 3, 4, 5, 6, 7][:num_days]
        self.days = num_days
        self.trailer_size = [1, 2, 3][:num_trailers]
        self.trailers = num_trailers
        self.p_d = [1, 0.8, 0.6, 0.7, 0.9, 0.2, 0.1][:num_days]
        self.side = size
        self.grid_cells_count = self.side*self.side
        self.truck_location_count = self.side*self.side
        self.destination_count = self.side*self.side
        self.state_size = self.grid_cells_count*self.days*self.trailers*(2**self.grid_cells_count)
        self.side_length = 1000

        self.all_actions = []
        for i in range(self.grid_cells_count):
            self.all_actions.append(i)

        self.num_possible_actions = len(self.all_actions)

        assert int(str(np.sqrt(self.grid_cells_count)).split('.')[1])==0, 'grid_cells_count must have an integer square root'

        self.top_row_states = list(range(self.grid_cells_count-int(np.sqrt(self.grid_cells_count)),self.grid_cells_count))
        self.bottom_row_states = list(range(int(np.sqrt(self.grid_cells_count))))
        self.left_column_states = list(range(0, self.grid_cells_count, int(np.sqrt(self.grid_cells_count))))
        self.right_column_states = list(range(int(np.sqrt(self.grid_cells_count))-1, self.grid_cells_count, int(np.sqrt(self.grid_cells_count))))

        if self.side == 2:
            self.b =  [0.70691977, 0.67605845, 0.95818857, 0.75984307]
        elif self.side == 3:
            self.b = [0.70691977, 0.67605845, 0.95818857, 0.75984307, 0.56210144,
           0.51252936, 0.53081853, 0.48575862, 0.65955056]
        self.load_probabilities = []

        for day in range(self.days):
            loads_matrix = np.zeros((self.grid_cells_count, self.grid_cells_count))
            for i in range(self.grid_cells_count):
                for j in range(self.grid_cells_count):
                    if i != j:
                        loads_matrix[i][j] = self.p_d[day]*self.b[i]*(1-self.b[j])

            self.load_probabilities.append(loads_matrix)

        #randomly assign loads to the grids
    def locate_loads(self, truck_location, day):
        load = []
        probabilities = self.load_probabilities[day - 1][truck_location]
        for i in range(self.grid_cells_count):
            number = np.random.choice(np.arange(0, 2), p = [1 - probabilities[i], probabilities[i]])
            load.append(number)
        return load

    def get_distance(self, truck_location,destination):
        distance = (abs(truck_location-destination)//self.side)*(self.side_length/self.side)+(abs(truck_location-destination)%self.side)*(self.side_length/self.side)
        return distance

    def get_rewards(self, current_state, current_action):
        truck_location = self.get_state_exp(current_state)[0]
        day = self.get_state_exp(current_state)[1]
        trailer_type = self.get_state_exp(current_state)[2]
        load = self.get_state_exp(current_state)[3]
        if current_action == truck_location:
            reward = 0
        elif load[current_action] == 1:
            if trailer_type == 1:
                reward = 1*self.get_distance(truck_location,current_action)*self.b[truck_location]
            elif trailer_type == 2:
                reward = 2*self.get_distance(truck_location,current_action)*self.b[truck_location]
            else:
                reward = 3*self.get_distance(truck_location,current_action)*self.b[truck_location]
        elif load[current_action] == 0:
            if trailer_type == 1:
                reward = -1*self.get_distance(truck_location,current_action)
            elif trailer_type == 2:
                reward = -2*self.get_distance(truck_location,current_action)
            else:
                reward = -3*self.get_distance(truck_location,current_action)
        return reward

    def get_actual_next_grid_state(self, action, current_state, is_test):
        day = self.get_state_exp(current_state)[1]
        trailer_type = self.get_state_exp(current_state)[2]
        next_truck_location = action
        next_day = day%self.days + 1
        next_trailer_type = trailer_type%self.trailers + 1
        new_load = self.locate_loads(next_truck_location, next_day)
        return next_truck_location, next_day, next_trailer_type, new_load

    def get_numeric_state_nomadic_trucker(self, truck_location, day, trailer_type, load):

        state_no = (truck_location)*(2**self.grid_cells_count) + (day-1)*(2**self.grid_cells_count)*self.grid_cells_count + (trailer_type-1)*(2**self.grid_cells_count)*self.grid_cells_count*self.days
        load_rev = reversed(load)
        j = 0
        for i in load_rev:
            state_no += i*(2**j)
            j+=1

        assert state_no>=0 and state_no< self.state_size, 'unknown state_no %s found'%state_no

        return state_no

    def get_state_exp(self, state):

        load = []
        b = state%(2**self.grid_cells_count)
        load_num = str(bin(b).replace("0b", ""))
        for i in load_num:
            load.append(int(i))
        if len(load) != self.grid_cells_count:
            indices = self.grid_cells_count - len(load)
            missing_indices = [0]*indices
            load = missing_indices + load
        a = state - b
        truck_location = (a%((2**self.grid_cells_count)*self.grid_cells_count))//(2**self.grid_cells_count)
        a = a - truck_location*(2**self.grid_cells_count)
        day = (a//((2**self.grid_cells_count)*self.grid_cells_count))%self.days + 1
        c = (a%((2**self.grid_cells_count)*self.grid_cells_count))//((2**self.grid_cells_count)*self.days)
        a = a - c*(2**self.grid_cells_count)*self.grid_cells_count*self.days
        trailer_type = a//((2**self.grid_cells_count)*self.grid_cells_count*self.days) + 1

        return truck_location, day, trailer_type, load

    def get_first_state(self):
        truck, dayofweek, trailer = (np.random.randint(0,self.grid_cells_count), np.random.randint(1, 1 + self.days), np.random.randint(1,1 + self.trailers))
        load = self.locate_loads(truck, dayofweek)
        return truck, dayofweek, trailer, load
