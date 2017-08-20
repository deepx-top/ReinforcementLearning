import enum
import random
import numpy as np


class q_action(enum.Enum):
    '''
    Since Env is 1-dimension,so only Left and Right available
    '''
    LEFT = 0
    RIGHT = 1


class q_env(object):
    """1-dimension environment for the simple example"""

    def __init__(self):
        super(q_env, self).__init__()
        self.state = '------T'
        self.bot_index = 0
        self.bot = 'o'
        self.obs = {}
        self.reward = 0
        self.reset()

    @property
    def state_num(self):
        return len(self.state)

    def reset(self):
        self.bot_index = 0
        self.obs['state'] = self.state
        self.obs['bot_index'] = self.bot_index
        self.reward = 0
        return self.obs, self.reward

    def step(self, act):
        t_state = list(self.state)
        if(act == q_action.LEFT):
            if self.bot_index > 0:
                self.bot_index = self.bot_index - 1
        else:
            if self.bot_index + 1 >= len(self.state):
                self.reward = 1
            else:
                self.bot_index = self.bot_index + 1
        self.obs['state'] = t_state
        self.obs['bot_index'] = self.bot_index

        return self.obs, self.reward


def q_table_learning():
    env = q_env()

    # Q_TABLE Q(s,a) is all zeros
    #        left right
    # state: x    x
    q_table = np.zeros([env.state_num, len(q_action)])

    # q_learning paraments
    EPOCHS = 13
    EXPLOR_RATE = 0.1
    Alpha = 0.1
    GAMMA = 0.9
    STEP_MAX = 100
    for eps in range(EPOCHS):
        obs, reward = env.reset()
        step = 0
        while step < STEP_MAX and not reward:
            step = step + 1
            curr_loc = obs['bot_index']
            vsa = q_table[curr_loc, :]
            if (random.random() < EXPLOR_RATE) or (vsa.all() == 0):
                action = random.choice([q_action.LEFT, q_action.RIGHT])
            else:
                action = q_action(vsa.argmax())
            obs, reward = env.step(action)
            new_loc = obs['bot_index']
            pred = q_table[curr_loc, action.value]
            if reward:
                target = reward
            else:
                target = reward + GAMMA * q_table[new_loc, :].max()
            q_table[curr_loc, action.value] += Alpha * (target - pred)
            curr_loc = new_loc
            if reward:
                env.reset()
        print '{epochs: %s; steps: %s}' % (eps,step)
    return q_table


def main():
    qtable = q_table_learning()
    print qtable


if __name__ == '__main__':
    main()
