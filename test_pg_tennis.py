import numpy as np

def sample(distribution, episode_no):
    # act = np.random.choice(ACTION_NUM, p=distribution)
    # return act
    epsilon = 0.2 * (1-1e-5 ** episode_no)
    rand = np.random.random()
    if rand < epsilon:
        act = np.random.randint(0, 18 - 1)
    else:
        act = np.argmax(distribution)
    print "##########"
    return act

dis = [9e-02,6.8e-01,1e-04,
       6e-03,1e-04,4e-04,
       5e-03,2e-04,7e-03,
       6e-03,2e-04,1.3e-01,
       4e-03,1e-04,2e-04,
       3e-04,4e-04,7e-02]

# x = sample(dis, 0.1)
print "sample: ", sample(dis, 0.1)

GAMMA = 0.99
rewards = [0,0,0,0,0,0,0,0,0,1]

def discount_rewards(rewards_):

    discounted_r = []
    running_add = 0
    reversed_rewards = rewards_
    reversed_rewards.reverse()
    # print "reversed_rewards: ", reversed_rewards
    for _, _reward in enumerate(reversed_rewards):
        running_add = running_add * GAMMA + _reward
        discounted_r.append(running_add)
    discounted_r.reverse()

    return discounted_r

discount_rewards(rewards)