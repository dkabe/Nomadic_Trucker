import numpy as np
from random import randint
import itertools
import pickle
from nomadic_trucker import nomadictrucker

#nomadictrucker(size, days, trailers)

model = nomadictrucker(2, 1, 1)
# model = nomadictrucker(2, 3, 3)
# model = nomadictrucker(2, 7, 3)


transition_probs = np.zeros((len(model['all_actions']), len(model['states']), len(model['states'])))
lst = list(itertools.product([0, 1], repeat=4))

def t_p():
    for action in model["all_actions"]:
        for load in lst:
            for new_load in lst:
                for truck_location in range(model['size']):
                    for day in range(1, (model['num_days']+1)):
                        for trailer in range(1, (model['num_trailers']+1)):
                            from_state = model['get_numeric_state_nomadic_trucker'](truck_location, day, trailer, load)
                            next_state = model['get_actual_next_grid_state'](action, from_state)
                            to_state = model['get_numeric_state_nomadic_trucker'](next_state[0], next_state[1], next_state[2], new_load)
                            transition_probs[model['all_actions'].index(action), from_state, to_state] = 1/(16)
    return transition_probs


transition_probs = t_p()

gamma = 0.90
theta = 0.0001
V_s = np.random.rand(len(model['states']))

delta = None
old_V_s = [0]*len(model['states'])
loop_count = 0
while delta is None or delta > theta:

    for state_index in model['states']:

        max_value = -float('inf')
        best_action = None
        new_V_s = 0

        for action in model['all_actions']:
            # immediate reward
            new_V_s = model['get_rewards'](state_index, action)

            # future reward
            for next_state_index, next_state in enumerate(model['states']):
                new_V_s += gamma * transition_probs[action, state_index, next_state_index]*V_s[next_state_index]


            if new_V_s > max_value:
                best_action = action
                max_value = new_V_s

        V_s[state_index] = max_value


    delta = max(abs(old_V_s - V_s))
    print('V_s %s delta %s'%(V_s[0],delta))
    old_V_s = V_s

    loop_count+=1
#print('\nloop_count %s'%loop_count)


pi_s = [None]*len(model['states'])
for state_index in model['states']:
    max_value = -float('inf')
    best_action = None
    new_V_s = 0

    for action_index, action in enumerate(model['all_actions']):
        new_V_s = model['get_rewards'](state_index, action)

        for next_state_index, next_state in enumerate(model['states']):
            new_V_s += gamma * transition_probs[action, state_index, next_state_index]*V_s[next_state_index]
        if new_V_s > max_value:
            best_action = action
            max_value = new_V_s

    pi_s[state_index] = best_action

with open('Trucker_V_2_'+str(days)+'.pickle', 'wb') as fp:
    pickle.dump(V_s, fp)
with open('Trucker_pi_2_'+str(days)+'.pickle', 'wb') as fp:
    pickle.dump(pi_s, fp)
