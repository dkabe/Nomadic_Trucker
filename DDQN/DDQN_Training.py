import numpy as np
import itertools
from DDQN_model_new import Environment, Agent

from collections import deque
import time
import pickle

def train_ddqn(episode,size,batch_size):

    #  initialize rewards, losses and episode lengths with zeros
    all_episode_rewards = np.zeros(episode)
    all_episode_losses = np.zeros(episode)
    all_episode_lengths = np.zeros(episode)

    # each episode represent a day
    for e in range(episode):

        loaded_moves = 0
        score = 0
        # generate a first state
        current_state = env.get_first_state()
        trailer_loc, day_of_week, trailer_size, load_vector = current_state # initial state

        next_state = None
        done = False

        for t in itertools.count():

            action = agent.act(current_state, num_actions)  # get action
            if load_vector[action] == 1:
                loaded_moves += 1

            current_state_encode = env.get_numeric_state_nomadic_trucker(trailer_loc, day_of_week, trailer_size, load_vector)
            reward = env.get_rewards(current_state_encode, action)

            # move to next state after taking action
            next_taxi_location, next_day, next_trailer_type, next_load_vector = env.get_actual_next_grid_state(action, current_state_encode, is_test)
            next_state = next_taxi_location, next_day, next_trailer_type, next_load_vector

            score += reward

            if loaded_moves == 10: # end iteration after 10 loaded moves
                done = True

            agent.remember(current_state, action, reward, next_state, done)
            current_state = next_state  # update current state
            trailer_loc, day_of_week, trailer_size, load_vector = current_state
            loss = agent.replay(num_actions) # calculate loss

            if done == True:

                agent.align_target_model()

                break

        agent.epsilon = max(agent.epsilon - (1 / episode), agent.epsilon_min)  # decrease epsilon step by step
        all_episode_rewards[e] = score
        all_episode_losses[e] = loss
        all_episode_lengths[e] = t
        print('episode',e,'is completed')

        if (e + 1) % 1000 == 0 and e != 0:
            print("episode: {}, score: {}, loss : {}".format(e, np.mean(all_episode_rewards[(e + 1) - 1000:e]),
                                                             np.mean(all_episode_losses[(e + 1) - 1000:e])))

    #  save the model
    agent.model.save(str(batch_size) +'weights'+str(size)+'.h5')
    return all_episode_losses, all_episode_rewards

#grid size
# grid = 2
grid = 3

# possible days
d = [1, 3, 7]

# possible trailer types
t = [1, 3, 3]

#Batch_size that are going to be trained for 3x3
Batch = [32, 64, 128]

#Batch_size that are going to be trained for 2x2
# Batch = [15, 32, 64]

for day, trailer in zip(d, t):
    size = grid
    days = day
    trailers = trailer
    num_actions = size*size
    state_inputs = 3 + num_actions
    state_space_size = (size*size)*day*trailers*(2**(size*size))
    print(state_space_size)
    for batch in b:
        batch_size = batch
        print(batch)
        agent = Agent(num_actions, state_inputs, batch_size, state_space_size)  # number of actions, number of state inputs (when state is flattened)
        env = Environment(agent, size, days, trailers)
        is_test = False
        start_time = time.time()
        all_episode_losses, all_episode_rewards = train_ddqn(10000,state_space_size,batch_size)  # give number of epoch as an argument
        end_time = time.time()
        print(end_time - start_time)
        with open(str(batch_size)+'DDQN_'+str(state_space_size)+'_loss_results.pickle', 'wb') as fp:
            pickle.dump(all_episode_losses, fp)
        with open(str(batch_size)+'DDQN_'+str(state_space_size)+'_rewards_results.pickle', 'wb') as fp:
            pickle.dump(all_episode_rewards, fp)
