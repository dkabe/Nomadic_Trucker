import numpy as np
from random import randint
import itertools

def nomadictrucker(size, days, trailers):
    # state is made up of four components: truck location, day of week, trailer type and load vector
    side = size
    days = days
    trailers = trailers
    
    
    days_of_week = [1, 2, 3, 4, 5, 6, 7][:days] # represents Monday - Sunday
  
    days = len(days_of_week)
    trailer_size = [1, 2, 3][:trailers] # represents small, medium large
    trailers = len(trailer_size) # number of different trailers
    p_d = [1, 0.8, 0.6, 0.7, 0.9, 0.2, 0.1][:days] # probabilities of loads arising each day (high during the week and low on weekends)
    grid_cells_count = side*side
    state_size = grid_cells_count*days*trailers*(2**grid_cells_count) # total number of unique states
    side_length = 1000

    # probabilities of loads generating out of each location:
    if side == 2:
        b =  [0.70691977, 0.67605845, 0.95818857, 0.75984307]
    elif side == 3:
        b = [0.70691977, 0.67605845, 0.95818857, 0.75984307, 0.56210144,
       0.51252936, 0.53081853, 0.48575862, 0.65955056]

    # possible actions given the state:
    all_actions = []
    for i in range(grid_cells_count):
        all_actions.append(i)

    num_possible_actions = len(all_actions)

    # hash (#) represents walls
    # pipe (|) represents cell divisions
    #
    # #####################
    # #  6  |  7   |  8   #
    # #  3  |  4   |  5   #
    # #  0  |  1   |  2   #
    # #####################

    assert int(str(np.sqrt(grid_cells_count)).split('.')[1])==0, 'grid_cells_count must have an integer square root'

    top_row_states = list(range(grid_cells_count-int(np.sqrt(grid_cells_count)),grid_cells_count))
    bottom_row_states = list(range(int(np.sqrt(grid_cells_count))))
    left_column_states = list(range(0, grid_cells_count, int(np.sqrt(grid_cells_count))))
    right_column_states = list(range(int(np.sqrt(grid_cells_count))-1, grid_cells_count, int(np.sqrt(grid_cells_count))))


    load_probabilities = []

    for day in range(days):
        loads_matrix = np.zeros((grid_cells_count, grid_cells_count))
        for i in range(grid_cells_count):
            for j in range(grid_cells_count):
                if i != j:
                    loads_matrix[i][j] = p_d[day]*b[i]*(1-b[j])

        load_probabilities.append(loads_matrix) # 7 load probability matrices, for each day of the week where each ijth entry represents the probability of a load arising
                                                # that will need to go from i to j

    # randomly assign loads to the grids based on the load probabilities
    def locate_loads(truck_location, day):
        load = []
        probabilities = load_probabilities[day - 1][truck_location]
        for i in range(grid_cells_count):
            number = np.random.choice(np.arange(0, 2), p = [1 - probabilities[i], probabilities[i]])
            load.append(number)
        return load

    # calculate distance between grid cells
    def get_distance(truck_location,destination):
        distance = (abs(truck_location-destination)//side)*(side_length/side)+(abs(truck_location-destination)%side)*(side_length/side)
        return distance

    # trailers move in cyclic order small --> medium --> large
    def get_next_trailer_type(current_trailer_type):
        trailer_type = current_trailer_type%trailers + 1
        return trailer_type

    # moving from one day to the next
    def get_next_day(current_day):
        day = current_day%days + 1
        return day

    # get trailer_type*distance*load_probability reward if you move loaded
    # get -trailer_type*distance reward if you move empty
    def get_rewards(current_state, current_action):
        truck_location = get_state_exp(current_state)[0]
        day = get_state_exp(current_state)[1]
        trailer_type = get_state_exp(current_state)[2]
        load = get_state_exp(current_state)[3]
        if current_action == truck_location:
            reward = 0
        elif load[current_action] == 1:
            if trailer_type == 1: # move loaded to a destination with 1 as index (load available FROM location TO destination)
                reward = 1*get_distance(truck_location,current_action)*b[truck_location]
            elif trailer_type == 2:
                reward = 2*get_distance(truck_location,current_action)*b[truck_location]
            else:
                reward = 3*get_distance(truck_location,current_action)*b[truck_location]
        elif load[current_action] == 0:
            if trailer_type == 1: # move loaded to a destination with 1 as index (load available FROM location TO destination)
                reward = -1*get_distance(truck_location,current_action)
            elif trailer_type == 2:
                reward = -2*get_distance(truck_location,current_action)
            else:
                reward = -3*get_distance(truck_location,current_action)
        return reward


    # at each move new loads appear
    # truck goes to destination cell or stays put if chooses stay action
    def get_actual_next_grid_state(action, current_state):
        day = get_state_exp(current_state)[1]
        trailer_type = get_state_exp(current_state)[2]
        next_truck_location = action
        next_day = get_next_day(day)
        next_trailer_type = get_next_trailer_type(trailer_type)
        new_load = locate_loads(next_truck_location, next_day)
        return next_truck_location, next_day, next_trailer_type, new_load

    # function to encode 4 states into 1 numeric state
    def get_numeric_state_nomadic_trucker(truck_location, day, trailer_type, load):

        state_no=(truck_location)*(2**grid_cells_count) + (day-1)*(2**grid_cells_count)*grid_cells_count + (trailer_type-1)*(2**grid_cells_count)*grid_cells_count*days
        load_rev=reversed(load)
        j=0
        for i in load_rev:
            state_no += i*(2**j)
            j+=1

        assert state_no>=0 and state_no< state_size, 'unknown state_no %s found'%state_no

        return state_no

    # function to decode numeric state into truck location, day, trailer, load
    def get_state_exp(state):

        load = []
        b = state%(2**grid_cells_count)
        load_num = str(bin(b).replace("0b", ""))
        for i in load_num:
            load.append(int(i))
        if len(load) != grid_cells_count:
            indices = grid_cells_count - len(load)
            missing_indices = [0]*indices
            load = missing_indices + load
        a = state - b
        truck_location = (a%((2**grid_cells_count)*grid_cells_count))//(2**grid_cells_count)
        a = a - truck_location*(2**grid_cells_count)
        day = (a//((2**grid_cells_count)*grid_cells_count))%days + 1
        c = (a%((2**grid_cells_count)*grid_cells_count))//((2**grid_cells_count)*days)
        a = a - c*(2**grid_cells_count)*grid_cells_count*days
        trailer_type = a//((2**grid_cells_count)*grid_cells_count*days) + 1

        return truck_location, day, trailer_type, load

    # initialize the first state
    def get_first_state():
        truck, dayofweek, trailer = (np.random.randint(0,grid_cells_count), np.random.randint(1,(days+1)), np.random.randint(1,(trailers+1)))
        load = locate_loads(truck, dayofweek)
        return get_numeric_state_nomadic_trucker(truck, dayofweek, trailer, load)

    return {
        'size' :grid_cells_count,
        'num_days' :days,
        'num_trailers' :trailers,
        'get_first_state':get_first_state,
        'get_distance': get_distance,
        'get_actual_next_grid_state':get_actual_next_grid_state,
        'states':list(range(state_size)),
        'all_actions':all_actions,
        'get_rewards': get_rewards,
        'get_numeric_state_nomadic_trucker':get_numeric_state_nomadic_trucker,
        'get_state_exp':get_state_exp,
        'num_possible_actions': num_possible_actions
        }
