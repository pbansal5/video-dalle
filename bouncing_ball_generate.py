"""Contains code for data set creation as well as live environments."""

import argparse
import pickle
import imageio
import numpy as np
import scipy as sc
import multiprocessing as mp
import h5py
from tqdm import tqdm

# from spriteworld import renderers as spriteworld_renderers
# from spriteworld.sprite import Sprite
import os

def norm(x):
    """Overloading numpys default behaviour for norm()."""
    if len(x.shape) == 1:
        _norm = np.linalg.norm(x)
    else:
        _norm = np.linalg.norm(x, axis=1).reshape(-1, 1)
    return _norm

class PhysicsEnv:
    """Base class for the physics environments."""

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1.,
                 init_v_factor=0, friction_coefficient=0., seed=None,
                 sprites=False):
        """Initialize a physics env with some general parameters.

        Args:
            n (int): Optional, number of objects in the scene.
            r (float)/list(float): Optional, radius of objects in the scene.
            m (float)/list(float): Optional, mass of the objects in the scene.
            hw (float): Optional, coordinate limits of the environment.
            eps (float): Optional, internal simulation granularity as the
                fraction of one time step. Does not change speed of simulation.
            res (int): Optional, pixel resolution of the images.
            t (float): Optional, dt of the step() method. Speeds up or slows
                down the simulation.
            init_v_factor (float): Scaling factor for inital velocity. Used only
                in Gravity Environment.
            friction_coefficient (float): Friction slows down balls.
            seed (int): Set random seed for reproducibility.
            sprites (bool): Render selection of sprites using spriteworld
                instead of balls.

        """
        np.random.seed(seed)

        self.n = n
        self.r = np.array([[r]] * n) if np.isscalar(r) else r
        self.m = np.array([[m]] * n) if np.isscalar(m) else m
        self.hw = hw
        self.internal_steps = granularity
        self.eps = 1 / granularity
        self.res = res
        self.t = t

        self.x = self.init_x()
        self.v = self.init_v(init_v_factor)
        self.a = np.zeros_like(self.v)

        self.fric_coeff = friction_coefficient
        self.v_rotation_angle = 2 * np.pi * 0.05

        if n > 3:
            self.use_colors = True
        else:
            self.use_colors = False

        if sprites:
            self.renderer = spriteworld_renderers.PILRenderer(
                image_size=(self.res, self.res),
                anti_aliasing=10,
            )

            shapes = ['triangle', 'square', 'circle', 'star_4']

            if not np.isscalar(r):
                print("Scale elements according to radius of first element.")

            # empirical scaling rule, works for r = 1.2 and 2
            self.scale = self.r[0] / self.hw / 0.6
            self.shapes = np.random.choice(shapes, 3)
            self.draw_image = self.draw_sprites

        else:
            self.draw_image = self.draw_balls

    def init_v(self, init_v_factor):
        """Randomly initialise velocities."""
        v = np.random.normal(size=(self.n, 2))
        v = v / np.sqrt((v ** 2).sum()) * .5
        return v

    def init_x(self):
        """Initialize ojbject positions without overlap and in bounds."""
        good_config = False
        while not good_config:
            x = np.random.rand(self.n, 2) * self.hw / 2 + self.hw / 4
            good_config = True
            for i in range(self.n):
                for z in range(2):
                    if x[i][z] - self.r[i] < 0:
                        good_config = False
                    if x[i][z] + self.r[i] > self.hw:
                        good_config = False

            for i in range(self.n):
                for j in range(i):
                    if norm(x[i] - x[j]) < self.r[i] + self.r[j]:
                        good_config = False
        return x

    def simulate_physics(self, actions):
        """Calculates physics for a single time step.

        What "physics" means is defined by the respective derived classes.

        Args:
            action (np.Array(float)): A 2D-float giving an x,y force to
                enact upon the first object.

        Returns:            factor = np.exp(- (((I - self.curtain_x[i, 0]) ** 2 +
                                (J - self.curtain_x[i, 1]) ** 2) /
                               (self.r[i] ** 2)) ** 4)
            d_vs (np.Array(float)): Velocity updates for the simulation.

        """
        raise NotImplementedError

    def step(self, action=None, mass_center_obs=False):
        """Full step for the environment."""
        if action is not None:
            # Actions are implemented as hardly affecting the first object's v.
            self.v[0] = action * self.t
            actions = True

        else:
            actions = False

        for _ in range(self.internal_steps):
            self.x += self.t * self.eps * self.v

            if mass_center_obs:
                # Do simulation in center of mass system.
                c_body = np.sum(self.m * self.x, 0) / np.sum(self.m)
                self.x += self.hw / 2 - c_body

            self.v -= self.fric_coeff * self.m * self.v * self.t * self.eps
            self.v = self.simulate_physics(actions)

        img = self.draw_image()
        state = np.concatenate([self.x, self.v], axis=1)
        done = False

        return img, state, done

    def get_obs_shape(self):
        """Return image dimensions."""
        return (self.res, self.res, 3)

    def get_state_shape(self):
        """Get shape of state array."""
        state = np.concatenate([self.x, self.v], axis=1)
        return state.shape

    @staticmethod
    def ar(x, y, z):
        """Offset array function."""
        return z / 2 + np.arange(x, y, z, dtype='float')

    def draw_balls(self):
        """Render balls on canvas."""
        if self.n > 3 and not self.use_colors:
            raise ValueError(
                'Must self.use_colors if self.n > 3.')

        if self.n > 6:
            raise ValueError(
                'Max self.n implemented currently is 6.')

        img = np.zeros((self.res, self.res, 3), dtype='float')
        [I, J] = np.meshgrid(self.ar(0, 1, 1. / self.res) * self.hw,
                             self.ar(0, 1, 1. / self.res) * self.hw)

        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]])

        for i in range(self.n):
            factor = np.exp(- (((I - self.x[i, 0]) ** 2 +
                                (J - self.x[i, 1]) ** 2) /
                               (self.r[i] ** 2)) ** 4)

            if self.use_colors:
                img[:, :, 0] += colors[i, 0] * factor
                img[:, :, 1] += colors[i, 1] * factor
                img[:, :, 2] += colors[i, 2] * factor

            else:
                img[:, :, i] += factor

        img[img > 1] = 1

        return img

    def draw_sprites(self):
        """Render sprites on the current locations."""

        s1 = Sprite(self.x[0, 0] / self.hw, 1 - self.x[0, 1] / self.hw,
                    self.shapes[0],
                    c0=255, c1=0, c2=0, scale=self.scale)
        s2 = Sprite(self.x[1, 0] / self.hw, 1 - self.x[1, 1] / self.hw,
                    self.shapes[1],
                    c0=0, c1=255, c2=0, scale=self.scale)
        s3 = Sprite(self.x[2, 0] / self.hw, 1 - self.x[2, 1] / self.hw,
                    self.shapes[2],
                    c0=0, c1=0, c2=255, scale=self.scale)

        sprites = [s1, s2, s3]
        img = self.renderer.render(sprites)

        return img / 255.

    def draw_curtain(self):
        """Render balls on canvas."""
        if self.n > 3 and not self.use_colors:
            raise ValueError(
                'Must self.use_colors if self.n > 3.')

        if self.n > 6:
            raise ValueError(
                'Max self.n implemented currently is 6.')

        img = np.zeros((self.res, self.res, 3), dtype='float')
        [I, J] = np.meshgrid(self.ar(0, 1, 1. / self.res) * self.hw,
                             self.ar(0, 1, 1. / self.res) * self.hw)

        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]])

        for i in range(self.n):
            factor = np.logical_and(
                np.abs(I - self.curtain_x[i, 0]) < 0.4,
                np.abs(J - self.curtain_x[i, 1]) < 0.4,
            ).astype(float)

            if self.use_colors:
                img[:, :, 0] += colors[i, 0] * factor
                img[:, :, 1] += colors[i, 1] * factor
                img[:, :, 2] += colors[i, 2] * factor

            else:
                img[:, :, i] += factor

        img[img > 1] = 1

        return img

    def reset(self, init_v_factor=None):
        """Resets the environment to a new configuration."""
        self.v = self.init_v(init_v_factor)
        self.a = np.zeros_like(self.v)
        self.x = self.init_x()


class BillardsEnv(PhysicsEnv):
    """Billiards or Bouncing Balls environment."""

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1.,
                 init_v_factor=0, friction_coefficient=0., seed=None, sprites=False):
        """Initialise arguments of parent class."""
        super().__init__(n, r, m, hw, granularity, res, t, init_v_factor,
                         friction_coefficient, seed, sprites)

        # collisions is updated in step to measure the collisions of the balls
        self.collisions = 0

    def simulate_physics(self, actions):
        # F = ma = m dv/dt ---> dv = a * dt = F/m * dt
        v = self.v.copy()

        # check for collisions with wall
        for i in range(self.n):
            for z in range(2):
                next_pos = self.x[i, z] + (v[i, z] * self.eps * self.t)
                # collision at 0 wall
                if not self.r[i] < next_pos:
                    self.x[i, z] = self.r[i]
                    v[i, z] = - v[i, z]
                # collision at hw wall
                elif not next_pos < (self.hw - self.r[i]):
                    self.x[i, z] = self.hw - self.r[i]
                    v[i, z] = - v[i, z]

        # check for collisions with objects
        for i in range(self.n):
            for j in range(i):

                dist = norm((self.x[i] + v[i] * self.t * self.eps)
                            - (self.x[j] + v[j] * self.t * self.eps))

                if dist < (self.r[i] + self.r[j]):
                    if actions and j == 0:
                        self.collisions = 1

                    w = self.x[i] - self.x[j]
                    w = w / norm(w)

                    v_i = np.dot(w.transpose(), v[i])
                    v_j = np.dot(w.transpose(), v[j])

                    if actions and j == 0:
                        v_j = 0

                    new_v_i, new_v_j = self.new_speeds(self.m[i], self.m[j], v_i, v_j)

                    v[i] += w * (new_v_i - v_i)
                    v[j] += w * (new_v_j - v_j)

                    if actions and j == 0:
                        v[j] = 0

        return v

    def new_speeds(self, m1, m2, v1, v2):
        """Implement elastic collision between two objects."""
        new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
        new_v1 = new_v2 + (v2 - v1)
        return new_v1, new_v2

    def step(self, action=None):
        """Overwrite step functino to ensure collisions are zeroed beforehand."""
        self.collisions = 0
        return super().step(action)


def generate_fitting_run(env_class, run_len=100, run_num=1000, max_tries=10000,
                         res=50, n=2, r=1., dt=0.01, gran=10, fc=0.3, hw=10,
                         m=1., seed=None,
                         init_v=None, check_overlap=False, sprites=False):
    """Generate runs for environments.

    Integrated error checks. Parameters as passed to environments.
    """

    if init_v is None:
        init_v = [0.1]

    good_counter = 0
    bad_counter = 0
    good_imgs = []
    good_states = []

    for _try in tqdm(range(max_tries)):
        _init_v = np.random.choice(init_v)
        # init_v is ignored for BillardsEnv
        env = env_class(n=n, r=r, m=m, hw=hw, granularity=gran, res=res, t=dt,
                        init_v_factor=_init_v, friction_coefficient=fc, seed=seed,
                        sprites=sprites)
        run_value = 0

        all_imgs = np.zeros((run_len, *env.get_obs_shape()))
        all_states = np.zeros((run_len, env.n, 4))

        run_value = 0
        for t in tqdm(range(run_len)):

            img, state, _ = env.step()
            all_imgs[t] = img
            all_states[t] = state

            run_value += np.sum(np.logical_and(
                state[:, :2] > 0, state[:, :2] < env.hw)) / (env.n * 2)

            if check_overlap:

                overlap = 0
                for i in range(n):
                    other = list(set(range(n)) - {i, })
                    # allow small overlaps
                    overlap += np.any(norm(state[i, :2] - state[other, :2])
                                      < 0.9 * (env.r[i] + env.r[other]))

                if overlap > 0:
                    run_value -= 1

        if run_value > (run_len - run_len / 100):
            good_imgs.append(all_imgs)
            good_states.append(all_states)
            good_counter += 1
        else:
            bad_counter += 1

        if good_counter >= run_num:
            break

    good_imgs = np.stack(good_imgs, 0)
    good_states = np.stack(good_states, 0)

    print(
        'Generation of {} runs finished, total amount of bad runs: {}. '.format(
            run_num, bad_counter))

    return good_imgs, good_states


def generate_data(save=True, test_gen=False, name='billiards', env=BillardsEnv,
                  config=None, output_dir="./data"):
    """Generate data for billiards or gravity environment."""

    num_runs = [10000, 300, 300] if (save and not test_gen) else [2, 5, 5]
    run_lens = [20, 100, 100]
    root_path = os.path.join(output_dir, name)
    os.makedirs(root_path, exist_ok=True)
    for run_types, run_num, run_len in zip(['train', 'val', 'test'], num_runs, run_lens):

        # generate runs
        X, y = generate_fitting_run(
            env, run_len=run_len, run_num=run_num, max_tries=10000, **config)

        # save data
        path = os.path.join(root_path, '{}.hdf5'.format(run_types))
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('y', data=y)
        print(f'File dumped to {path}')

    # also generate gif of data
    first_seq = (255 * X[:20].reshape(
        (-1, config['res'], config['res'], 3))).astype(np.uint8)
    imageio.mimsave(os.path.join(root_path, '{}.gif'.format(name)), first_seq, fps=24)



if __name__ == '__main__':
    """Create standard collection of data sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-gen', dest='test_gen', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.add_argument('--output-dir', type=str, default='./data', help='path to put files')
    args = parser.parse_args()

    config = {'res': 64, 'hw': 10, 'n': 1, 'dt': 1, 'm': 1., 'fc': 0,
              'gran': 2, 'r': 1.0, 'check_overlap': False}
    
    generate_data(
        save=args.save, test_gen=args.test_gen, name='billiards-curtain-elevator-reflector-n1-T20-gap',
        env=BillardsCurtainReflectorEnv, config=config, output_dir=args.output_dir)