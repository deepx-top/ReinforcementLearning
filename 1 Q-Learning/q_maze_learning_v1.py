# -*- coding:utf-8 -*-

import numpy as np
import pygame
from pygame.locals import *
import collections
import random
import time
import six


class Action(collections.namedtuple('Action', ["idx", "name", "movement"])):
    """Represents a signal action"""
    __slots__ = ()

    @classmethod
    def spec(cls, idx, name, movement):
        """Create an action to be used in ValidActions."""
        return cls(idx, name, movement)


class Actions(collections.namedtuple('Actions', ['N', 'E', 'S', 'W'])):
    ''' The Actions
    '''
    __slots__ = ()

    def __new__(cls, **kwargs):
        act = {}
        for name, (idx, movement) in six.iteritems(kwargs):
            act[name] = Action(
                idx=Actions._fields.index(name),
                name=name,
                movement=movement)
        return super(Actions, cls).__new__(cls, **act)


class EnvMaze(object):
    """ The environment in RL
        when an agent action in the environment,
        return reward, observise
    """

    def __init__(self):
        '''
        Initalize a environment
        the maze is an example from CS294 lecture 1
        '''
        super(EnvMaze, self).__init__()
        pygame.init()
        pygame.display.set_caption("EnvMaze")
        self._create_maze()
        self.obs = {}
        self.reward = 0
        # define actions
        self.actions = Actions(N=(0, (-1, 0)),
                               E=(1, (0, 1)),
                               S=(2, (1, 0)),
                               W=(3, (0, -1)))
        pygame.display.update()

    def _create_maze(self):
        # init screen
        self.screen_width = 640
        self.screen_height = 640
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), 0, 32)
        # maze attr define
        self.color_background = [255, 255, 255]
        self.screen.fill(self.color_background)
        # rend maze
        self.bot = pygame.image.load('./bot.jpg').convert()
        self.bot_val = 10
        self.maze_value = self._get_maze_value()
        # the state space
        self.loc2index = self._loc2index()
        self._render_maze()

    def _maze_attr(self, type):
        '''
        '''
        if type == 'COLOR':
            color_wall, color_start, color_goal = [
                0, 0, 0], [0, 255, 0], [255, 0, 0]
            return (color_wall, color_start, color_goal)
        if type == 'VALUE':
            val_wall, val_start, val_goal = 1, 2, 3
            return (val_wall, val_start, val_goal)
        raise 'type must be in [COLOR,VALUE]'

    def _get_maze_value(self, row=8, col=8):
        # prepare maze value
        # background_value=0
        maze_value = np.zeros([row, col])
        val_wall, val_start, val_goal = self._maze_attr('VALUE')
        # define the wall
        maze_value[:, 0], maze_value[0, :] = val_wall, val_wall
        maze_value[row - 1, :], maze_value[:, col - 1] = val_wall, val_wall

        loc_wall = [(2, 2), (2, 3), (2, 5), (3, 3), (3, 4),
                    (4, 1), (4, 4), (4, 6), (5, 2), (5, 4), (6, 5)]
        for _, (i, j) in enumerate(loc_wall):
            maze_value[i, j] = val_wall

        # define start
        self.loc_start = [(2, 0)]
        for _, (i, j) in enumerate(self.loc_start):
            maze_value[i, j] = val_start

        # define goal
        self.loc_goal = [(6, 7)]
        for _, (i, j) in enumerate(self.loc_goal):
            maze_value[i, j] = val_goal
        return maze_value

    def _render_maze(self):
        ''' render the color
        '''
        color_wall, color_start, color_goal = self._maze_attr('COLOR')
        val_wall, val_start, val_goal = self._maze_attr('VALUE')
        row_num, col_num = self.maze_value.shape[0], self.maze_value.shape[1]
        self.row_num, self.col_num = row_num, col_num
        self.row_step = self.screen_width / col_num
        self.col_step = self.screen_height / row_num
        # render the grid color
        # self.maze_value mapped to Rectangular Coordinates,need transposition
        # so the row_index is mapped to column and col_index is mapped to row
        for i in range(row_num):
            for j in range(col_num):
                if self.maze_value[i][j] == val_goal:
                    self._rend_loc((i, j), color_goal)
                elif self.maze_value[i][j] == val_start:
                    self._rend_loc((i, j), color_goal)
                elif self.maze_value[i][j] == val_wall:
                    self._rend_loc((i, j), color_wall)
        # rendr grid
        for i in range(col_num):
            pygame.draw.line(self.screen, color_wall, (0, self.col_step * i),
                             (self.screen_width, self.col_step * i), 1)
        for j in range(row_num):
            pygame.draw.line(self.screen, color_wall, (self.row_step * j, 0),
                             (self.row_step * j, self.screen_height), 1)

    def _rend_loc(self, loc, color):
        x, y = loc
        pygame.draw.rect(self.screen, color,
                         Rect([self.row_step * y + 1, self.col_step * x + 1],
                              [self.row_step - 1, self.col_step - 1]))
        pygame.display.update()

    def reset(self):
        self.bot_loc = self.loc_start
        self.screen.blit(self.bot, self._cvt2center(self.bot_loc))
        self.obs['bot_loc'] = self.bot_loc
        self.reward = 0
        pygame.display.update()
        return self.obs, self.reward

    def _cvt2center(self, loc):
        return [self.row_step * loc[0][1] + int(self.row_step / 4),
                self.col_step * loc[0][0] + int(self.col_step / 4)]

    @property
    def state_num(self):
        return (np.sum(self.maze_value == 0) + 2)

    def cvt_loc2index(self, loc):
        x, y = loc[0]
        return self.loc2index[10 * x + y]

    def get_loc2index(self):
        return self.loc2index

    def _loc2index(self):
        '''
        Since we got bot_loc,which need to conver to q_table's space index
        '''
        x, y = np.where(self.maze_value == 0)
        locs = {}
        loc_set = set(x * 10 + y)
        # add_start
        xs, ys = self.loc_start[0]
        loc_set.add(xs * 10 + ys)
        # add goal
        xg, yg = self.loc_goal[0]
        loc_set.add(xg * 10 + yg)
        for i, loc in enumerate(loc_set):
            locs[loc] = i
        return locs

    def actions(self):
        return self.actions

    def ava_action(self, loc):
        ''' the avaiable actions when robot in this location
        '''
        loc_x, loc_y = loc[0]
        ava_action = []
        for idx in range(len(self.actions)):
            x, y = self.actions[idx].movement
            nx, ny = loc_x + x, loc_y + y
            if nx in range(self.row_num) and ny in range(self.col_num):
                # the goal_val =3 is also need consider
                if (self.maze_value[nx, ny] == 0) or (self.maze_value[nx, ny] == 3):
                    ava_action.append(idx)
        return ava_action

    def step(self, action_idx):
        assert action_idx < len(self.actions)
        x, y = self.actions[action_idx].movement
        bot_loc_new = [(self.bot_loc[0][0] + x,
                        self.bot_loc[0][1] + y)]
        self._rend_loc((self.bot_loc[0][0], self.bot_loc[0][1]),
                       [255, 255, 255])
        self.screen.blit(self.bot, self._cvt2center(bot_loc_new))
        self.bot_loc = bot_loc_new
        pygame.display.update()

        self.obs['bot_loc'] = self.bot_loc
        if self.is_goal(self.bot_loc):
            self.reward = 1
        pygame.display.update()
        # time.sleep(0.1)
        return self.obs, self.reward

    def is_goal(self, loc):
        x, y = loc[0]
        xg, yg = self.loc_goal[0]
        if x == xg and y == yg:
            return 1
        return 0

    def stop(self):
        pygame.quit()


def q_maze_learning():
    env = EnvMaze()

    # Q_TABLE Q(s,a) is all zeros
    #        left right
    # state: x    x
    state_num,action_num=env.state_num,len(env.actions)
    q_table = np.zeros([state_num,action_num])

    # q_learning paraments
    EPOCHS = 30
    EXPLOR_RATE = 0.1
    Alpha = 0.1
    GAMMA = 0.9
    STEP_MAX = 300

    for eps in range(EPOCHS):
        obs, reward = env.reset()
        step = 0
        while step < STEP_MAX and not reward:
            if eps==12:
                mm=0
            step = step + 1
            curr_loc = obs['bot_loc']
            vsa = q_table[env.cvt_loc2index(curr_loc), :]
            if (random.random() < EXPLOR_RATE) or (np.sum(vsa==0) == action_num ):
                action = random.choice(env.ava_action(curr_loc))
            else:
                action = env.actions[vsa.argmax()].idx
            obs, reward = env.step(action)
            new_loc = obs['bot_loc']
            pred = q_table[env.cvt_loc2index(curr_loc), action]
            cx, cy = curr_loc[0]
            if reward:
                target = reward
            else:
                target = reward + GAMMA * \
                    q_table[env.cvt_loc2index(new_loc), :].max()
            q_table[env.cvt_loc2index(
                curr_loc), action] += Alpha * (target - pred)
            curr_loc = new_loc

        print '{epochs: %s; steps: %s}' % (eps, step)
    print env.get_loc2index()
    return q_table


def main():
    q_table = q_maze_learning()
    print q_table

if __name__ == '__main__':
    main()
