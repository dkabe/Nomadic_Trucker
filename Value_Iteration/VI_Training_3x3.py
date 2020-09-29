import numpy as np
from random import randint
import itertools
import pickle
from nomadic_trucker import nomadictrucker

model = nomadictrucker(3, 1, 1)
# model = nomadictrucker(3, 3, 3) #does not converge
# model = nomadictrucker(3, 7, 3) #does not converge


def get_transition_probs(action, from_state, to_state):
    from_state_day = model['get_state_exp'](from_state)[1]
    from_state_trailer = model['get_state_exp'](from_state)[2]
    to_state_location = model['get_state_exp'](to_state)[0]
    to_state_day = model['get_state_exp'](to_state)[1]
    to_state_trailer = model['get_state_exp'](to_state)[2]
    if (to_state_location == action and to_state_day == (from_state_day%(model['num_days']) + 1) and to_state_trailer == (from_state_trailer%(model['num_trailers']) + 1)):
        return 1/512
    else:
        return 0


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
                new_V_s += gamma * get_transition_probs(action, state_index, next_state_index)*V_s[next_state_index]


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
            new_V_s += gamma * get_transition_probs(action, state_index, next_state_index)*V_s[next_state_index]
        if new_V_s > max_value:
            best_action = action
            max_value = new_V_s

    pi_s[state_index] = best_action
    

with open('Trucker_V_3_'+str(days)+'.pickle', 'wb') as fp:
    pickle.dump(V_s, fp)
with open('Trucker_pi_3x3_'+str(days)+'.pickle', 'wb') as fp:
    pickle.dump(pi_s, fp)

