import gym
from gym import logger
from gym.envs.classic_control.mountain_car import *
import math
import numpy as np
from scipy.misc import derivative


class CartPoleSwingUp(gym.Wrapper):
    def __init__(self, env=gym.make('CartPole-v1'), **kwargs):
        super(CartPoleSwingUp, self).__init__(env, **kwargs)
        self.theta_dot_threshold = 4*np.pi

    def reset(self):
        self.env.env.state = [0, 0, np.pi, 0] + super().reset()
        return np.array(self.env.env.state)

    def step(self, action):
        state, reward, done, _ = super().step(action)
        self.env.env.steps_beyond_done = None
        x, x_dot, theta, theta_dot = state
        theta = (theta+np.pi)%(2*np.pi)-np.pi
        self.env.env.state = [x, x_dot, theta, theta_dot]
        
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta_dot < -self.theta_dot_threshold \
               or theta_dot > self.theta_dot_threshold
        
        if done:
            # game over
            reward = -10.
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            elif self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        else:
            if -self.theta_threshold_radians < theta and theta < self.theta_threshold_radians:
                # pole upright
                reward = 1.
            else:
                # pole swinging
                reward = 0.

        return np.array([x, x_dot, theta, theta_dot]), reward, done, {}

# Custom MC 

class CustomMountainCarEnv(MountainCarEnv):
    def __init__(self, goal_velocity=0):
        super().__init__(goal_velocity=goal_velocity)
        self.max_position = 2
        self.min_position = -self.max_position
        x = (self.max_position+self.min_position)/2
        goal_width=0.1
        self.min_goal_position= x-goal_width/2
        self.max_goal_position= x+goal_width/2
        
        self.out_score = -1000 # -1/(1-self.gamma)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + self._weight(position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity

        reward, done = self._reward_done(position)

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        # self.state = np.array([self.np_random.uniform(low=-0.9, high=-1.1), 0])
        # self.state = np.array([self.np_random.uniform(low=-0.1, high=0.1), 0])
        self.state = np.array([-1, 0])
        return np.array(self.state)

    def _height(self, x):
        # return (np.sin(7 * (x + 0.55)) * .45 + .55)*np.exp(0.1*x)+0.1*x
        return np.cos(np.pi * x) * .45 + .55 #+0.05*(x-self.min_position)
    
    def _weight(self, x):
        return derivative(self._height, x, dx=1e-6) /1.35
    
    def _reward_done(self, x):
#         x = np.clip(x, self.min_position, self.max_position)
        if x <= self.min_position or x >= self.max_position:
            return self.out_score, True
        elif x >= self.min_goal_position and x <= self.max_goal_position:
            return 0, False
        else:
            return -1, False
    
    def render(self, mode='human'): # pragma: no cover
        screen_width = 600
        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            screen_height = 400

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            carwidth = 40
            carheight = 20

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (-self.min_position) * scale
            flagy1 = self._height(0) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(self._weight(pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')