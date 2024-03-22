#!/usr/bin/env python
# coding=utf-8
import gym
import random
import time
import numpy as np
import pygame
from pygame import gfxdraw
from gym.envs.classic_control import rendering


class DeliverEnv(gym.Env):
    def __init__(self):
        self.state = (0, 0)
        self.done = None
        self.width, self.height = 5, 5
        self.goods = [(1, 1), (4, 3)]
        self.window = None
        self.clock = None
        self.viewer = None

    def reset(self):
        self.state = (0, 0)
        self.goods = [(1, 1), (4, 3)]
        self.done = None
        self.window = None
        self.clock = None
        self.viewer = None

    def step(self, action):
        # 动作格式：action
        x = action[0]
        y = action[1]
        # print('x: {}, y: {}'.format(x, y))
        if not (0<=self.state[0]+x <self.width and 0<=self.state[1]+y <self.height):
            done = True
            reward = -10
            # print('done')
        else:
            # print('x: {}, y: {}'.format(x, y))
            self.state = (self.state[0]+x, self.state[1]+y)
            if self.state in self.goods:
                self.goods.remove(self.state)
                reward = 10
            else:
                reward = -1
            done = self.judgeEnd()
        # 报告
        info = {}
        return self.state, reward, done, info

    def judgeEnd(self):
        if len(self.goods) == 0:
            return True
        return False

    def render(self, mode='human'):
        width = 500
        height = 500
        gap = 100
        # if self.window is None:
        #     pygame.init()
        #     pygame.display.init()
        #     self.screen = pygame.display.set_mode((width, height))
        # if self.clock is None:
        #     self.clock = pygame.time.Clock()
        if self.viewer is None:
            # print('coming')
            self.viewer = rendering.Viewer(width, height)
            for x in range(6):
                line = rendering.Line((x*gap, 0), (x*gap, height))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)
            for y in range(6):
                line = rendering.Line((0, y * gap), (width, y * gap))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)
            for item in self.goods:
                circle = rendering.make_circle(30)
                circle.set_color(135 / 255, 206 / 255, 250 / 255)  # blue
                move = rendering.Transform(translation=(item[0]*gap+gap//2, item[1]*gap+gap//2))
                circle.add_attr(move)
                self.viewer.add_geom(circle)

            self.robot = rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(1, 0.8, 0)
            self.viewer.add_geom(self.robot)
        # 设置机器人当前位置
        self.robotrans.set_translation(self.state[0]*gap+gap//2, self.state[1]*gap+gap//2)


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        self.viewer.close()



