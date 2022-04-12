from multiprocessing import AuthenticationError
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
    # Stores the player index, static
    player_index: Dict[Player, int] = field(init=False)
    # Records the player index corresponding to the game, mutable throughout the game
    state_index: np.array = field(init=False)
    # Records if the player is still alive
    alive: List[Boolean] = field(init=False)
    state: np.array = field(init=False)

    G: np.float64 = 1
    M: np.float64 = 1
    m: np.float64 = 0.1
    t: np.float64 = 0.0
    dt: np.float64 = 0.01
    steps: int = 10
    thrust_cap: np.float64 = 0.01
    min_radius: np.float64 = 0.1
    max_radius: np.float64 = 2.0

    thrust: np.array = field(init=False)
    modified: List[Boolean] = field(init=False)
    rank: int = field(init=False)
    ranks: np.array = field(init=False)

    def __post_init__(self):
        n = len(self.players)
        self.alive = [True] * n

        self.player_index = {player: index for index,
                             player in enumerate(self.players)}
        self.state_index = np.arange(n)

        pos = np.linspace(0, 2 * np.pi, n, endpoint=False)
        assert len(pos) == n
        self.state = np.hstack(
            (np.cos(pos), np.sin(pos), -np.sin(pos), np.cos(pos))).T
        assert self.state.shape == (n, 4)

        self.thrust = np.zeros((n, 2), dtype=np.float64)
        self.modified = [False] * n

        self.rank = n + 1
        self.ranks = np.array([0] * n, dype=int)

    def _diff_eq(self, state: np.array, thrust: np.array):
        '''
        Output the states based on Euler-Cromer Method
        '''
        r, v = state[:, :2], state[:, 2:]
        force = self.G * self.M * self.m * r / \
            np.power(np.linalg.norm(r, axis=1), 3).reshape((-1, 1)) + thrust
        v = v + force * self.dt / self.m
        r = r + v * self.dt
        t = t + self.dt

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

    def get_player_index(self, player: Player):
        return self.player_index[player]

    def get_state_index(self, player: Player):
        player_index = self.player_index[player]
        if self.alive[player_index]:
            return {
                'index': self.state_index[self.get_player_index(player)],
                'alive': True
            }
        else:
            return {'alive': False, 'rank': self.ranks[player_index]}

    def get_state(self):
        return {
            'state_index': self.state_index.tolist(),
            'alive': self.alive,
            'state': self.state.tolist(),
            'names': {index: player.name for index, player in enumerate(self.players)}
        }

    def get_player_state(self, player: Player):
        return {**self.get_state_index(player), **self.get_state()}

    def game_end(self, winners: List[int]):
        return {'end': True, 'winners': winners}

    def game_step(self):
        thrust = self.thrust
        for _ in range(self.steps):
            self._diff_eq(thrust)

            # Out players that are out of bounds
            norms = np.linalg.norm(self.state[:, :2], axis=1)
            inbounds = ((norms >= self.min_radius) |
                        (norms <= self.min_radius))
            num_winners = np.count_nonzero(inbounds)
            num_losers = len(inbounds) - num_winners
            if np.count_nonzero(inbounds) == 0:
                return self.game_end(np.arange(len(thrust))[self.alive])

            # Update the rank of each player
            losers = self.alive[:]
            losers[losers] = ~inbounds
            self.rank = self.rank - num_losers
            self.ranks[losers] = self.rank

            # Update alive so that players out of bounds are marked as dead
            self.alive[self.alive] = inbounds
            self.state_index[self.alive] = np.arange(num_winners)
            self.state_index[~self.alive] = -1
            self.state = self.state[self.alive]

            return {'end': False}
