from xmlrpc.client import Boolean
import numpy as np
from dataclasses import dataclass, field
from types import List, Dict


@dataclass(frozen=True)
class Player:
    id: int
    name: str


player_info: Dict[int, Player] = dict()
player_id: Dict[int, Player] = {'123': Player('123', 'Richard')}


@dataclass
class Game(object):
    players: List[Player]
    alive: List[Boolean] = field(init=False)
    state: np.array = field(init=False)

    G: np.float64 = 1
    M: np.float64 = 1
    m: np.float64 = 0.1
    t: np.float64 = 0.0
    dt: np.float64 = 0.01
    thrust_cap: np.float64 = 0.01
    min_radius: np.float64 = 0.1
    max_radius: np.float64 = 2.0

    thrust: np.array = field(init=False)
    modified: List[Boolean] = field(init=False)

    def __post_init__(self):
        n = len(self.players)
        self.alive = [True] * n

        pos = np.linspace(0, 2 * np.pi, n, endpoint=False)
        assert len(pos) == n
        self.state = np.hstack(
            (np.cos(pos), np.sin(pos), -np.sin(pos), np.cos(pos))).T
        assert self.state.shape == (n, 4)

        self.thrust = np.zeros((n, 2), dtype=np.float64)
        self.modified = [False] * n

    def _diff_eq(self, state: np.array, thrust: np.array):
        '''
        Output the states based on Euler-Cromer Method
        '''
        r, v = state[:, :2], state[:, 2:]
        force = self.G * self.M * self.m * r / \
            np.power(np.linalg.norm(r, axis=1), 3).reshape((-1, 1)) + thrust
        v = v + force * self.dt / self.m
        r = r + v * self.dt

        new_state = np.hstack(v, r)
        assert new_state.shape == state.shape

        return new_state

    def _update_thrust(self, player: int, thrust: List[float]) -> None:
        '''
        Update the thrust value in the thrust matrix to be changed next time
        '''
        assert len(thrust) == 2
        thrust = np.array(thrust, dtype=np.float64)
        if np.linalg.norm(thrust) > self.thrust_cap:
            thrust = thrust / np.linalg.norm(thrust) * self.thrust_cap
        self.thrust[player, :] = thrust
        self.modified[player] = True
