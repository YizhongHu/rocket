from multiprocessing import AuthenticationError
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import matplotlib.pyplot as plt

import string
import random


@dataclass(frozen=True)
class Player:
    id: int
    name: str


player_info: Dict[int, Player] = dict()
player_id: Dict[int, Player] = {'123': Player(
    '123', 'Richard'), '456': Player('456', 'Yizhong'), '789': Player('789', 'David')}


@dataclass
class Game(object):
    '''Represents the multiplayer game.

    Handles progressing the simulation, keeps track of each player's alive state,
    and give current player/game state based on request.

    Attributes:
        players: A List of players, the index corresponding to their in-game identification number
        player_index: Inverse mapping of players to their in-game identification number
        state_index: For each player, records their current state in the state vector based on player index
        alive: A List of Booleans, recording if each player is alive or not
        state: A numpy array recording each living player's current position and velocity. Each row
               corresponds to a player. The first two floats records the cartesian position, and the
               last two floats represent the cartesian velocity
        rank: A the rank of the player(s) that just lost
        ranks: A record of the ranks of each player. If the player is not out yet or the game did not end,
               the rank is set to 0

        G: Gravitational Constant, default: 1
        M: Mass of center object, default: 1
        m: Mass of each rocket, default: 0.01
        t: Time passed in simulation
        repel: the magnitude of the repelling force
        repel_power: the power of the repelling force
        dt: Simulation time step length

        steps: Number of simulation steps (dt) for each state update
        thrust_cap: The maximum thrust allowed, default: 0.01
        min_radius: The inside bound
        max_radius: The outside bound

        thrust: The newest thrust value requested by the players alive, indexed by state index
        modified: If the thrust value has been modified since the last state update, indexed by state index
    '''
    players: List[Player]
    player_index: Dict[Player, int] = field(init=False)
    state_index: np.array = field(init=False)
    alive: np.array = field(init=False)
    state: np.array = field(init=False)
    rank: int = field(init=False)
    ranks: np.array = field(init=False)

    G: np.float64 = 1
    M: np.float64 = 1
    m: np.float64 = 0.1
    t: np.float64 = 0.0
    repel: np.float64 = 1e-6
    repel_power: int = -4
    dt: np.float64 = 0.001

    steps: int = 100
    thrust_cap: np.float64 = 0.01
    min_radius: np.float64 = 0.1
    max_radius: np.float64 = 2.0

    thrust: np.array = field(init=False)
    modified: List[bool] = field(init=False)

    def __post_init__(self):
        # Initialize alive
        n = len(self.players)
        self.alive = np.array([True] * n)

        # Initialize index mappings
        self.player_index = {player: index for index,
                             player in enumerate(self.players)}
        self.state_index = np.arange(n)

        # Initialize game state
        pos = np.linspace(0, 2 * np.pi, n, endpoint=False)
        assert len(pos) == n
        self.state = np.vstack(
            (np.cos(pos), np.sin(pos), -np.sin(pos), np.cos(pos))).T
        assert self.state.shape == (n, 4)

        # Initialize thrust settings
        self.thrust = np.zeros((n, 2), dtype=np.float64)
        self.modified = [False] * n

        # Initialize rank records
        self.rank = n + 1
        self.ranks = np.array([0] * n, dtype=int)

    def _diff_eq(self, state: np.array, thrust: np.array):
        '''
        Output the states based on Euler-Cromer Method
        '''
        r, v = state[:, :2], state[:, 2:]

        d = np.linalg.norm(r, axis=1).reshape((-1, 1))
        center_force = -self.G * self.M * self.m * r / np.power(d, 3) + thrust

        x = np.expand_dims(np.subtract.outer(r[:, 0], r[:, 0]), axis=2)
        y = np.expand_dims(np.subtract.outer(r[:, 1], r[:, 1]), axis=2)
        rij = np.concatenate((x, y), axis=-1)
        dij = np.expand_dims(np.linalg.norm(rij, axis=-1) + 1e-6, axis=2)
        cross_force = np.sum(-self.G * self.M * self.m * rij *
                             (np.power(dij, -3) + self.repel * np.power(dij, self.repel_power - 1)), axis=1)

        assert cross_force.shape == center_force.shape

        force = center_force + cross_force

        v = v + force * self.dt / self.m
        r = r + v * self.dt
        self.t = self.t + self.dt
        new_state = np.hstack((r, v))
        assert new_state.shape == state.shape

        self.state = new_state

    def _update_thrust(self, player: Player, thrust: List[float]) -> None:
        '''
        Update the thrust value in the thrust matrix to be changed next time
        '''
        assert len(thrust) == 2
        thrust = np.array(thrust, dtype=np.float64)
        player = self.state_index[self.player_index[player]]
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

    def is_alive(self, player: Player):
        return self.alive[self.player_index[player]]

    def get_state(self):
        return {
            'state_index': self.state_index.tolist(),
            'alive': self.alive,
            'state': self.state.tolist(),
            'names': {self.state_index[index]: player.name
                      for index, player in enumerate(self.players) if self.alive[index]},
            'all_names': {index: player.name for index, player in enumerate(self.players)},
        }

    def get_player_state(self, player: Player):
        return {**self.get_state_index(player), **self.get_state()}

    def game_end(self, winners: List[int]):
        player_indices = []
        for winner in winners:
            winner_player_index = np.argmax(self.state_index == winner)
            player_indices.append(winner_player_index)
        self.ranks[player_indices] = 1
        return {'end': True, 'winners': player_indices, 'ranks': self.ranks}

    def game_step(self):
        thrust = self.thrust

        for _ in range(self.steps):
            self._diff_eq(self.state, thrust)

            # Out players that are out of bounds
            norms = np.linalg.norm(self.state[:, :2], axis=1)
            inbounds = ((norms >= self.min_radius) &
                        (norms <= self.max_radius))
            num_winners = np.count_nonzero(inbounds)
            num_losers = len(inbounds) - num_winners

            # Update the rank of each player
            losers = np.copy(self.alive)
            losers[losers] = ~inbounds
            self.rank = self.rank - num_losers
            self.ranks[losers] = self.rank

            # Update alive so that players out of bounds are marked as dead
            self.alive[self.alive] = inbounds
            self.state_index[self.alive] = np.arange(num_winners)
            self.state_index[~self.alive] = -1
            self.state = self.state[inbounds]
            thrust = thrust[inbounds]

            if num_winners == 0 or num_winners == 1:
                return self.game_end(np.arange(len(inbounds))[inbounds])

        self.thrust = thrust

        return {'end': False}


def generate_random_players(num_players: int, len_id: int = 10, len_name: int = 10):
    alphanums = string.ascii_letters + string.digits
    players = [''.join(random.choice(alphanums) for _ in range(len_id))
               for _ in range(num_players)]
    with open('names.txt', 'r') as file:
        all_names = file.readlines()
    names = [random.choice(all_names)[:-1] for _ in range(num_players)]

    return [Player(player, name) for player, name in zip(players, names)]


if __name__ == '__main__':
    players: List[Player] = generate_random_players(4)
    print(players)
    record: Dict[Player, List] = {player: list() for player in players}
    game = Game(players)
    for _ in range(100):
        for player in players:
            state = game.get_player_state(player)
            record[player].append(state['state'][state['index']])

        print(game.game_step())

    for player, path in record.items():
        states = np.array(path)
        plt.plot(states[:, 0], states[:, 1], label=player.name)

    plt.legend()
    plt.show()


# Players cannot control a dead rocket
# Players cannot access each others' ids
# Correct messages are sent to each rocket
