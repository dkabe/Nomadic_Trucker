import matplotlib.pyplot as plt

def plot_qlearning_stats(Q, all_episode_rewards, all_episode_lengths):
    plt.subplot(2,1,1)
    plt.plot(all_episode_rewards)
    plt.xlabel('Episode number')
    plt.ylabel('Collected reward')

    plt.subplot(2,1,2)
    plt.plot(all_episode_lengths)
    plt.xlabel('Episode number')
    plt.ylabel('Loss')

    plt.show()
