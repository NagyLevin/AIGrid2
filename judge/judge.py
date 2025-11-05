import socket
import argparse
import time
import json
import contextlib
import network
from pprint import pprint
import numpy as np

from typing import Any, Optional, Callable, NamedTuple, Literal

PlayerInput = Any

#: number of strikes before the player is disqualified (communications stop)
PLAYER_MAX_STRIKES = 5

class EnvironmentBase:
    """
    Concrete environments should subclass these, implementing ``reset``,
    ``next_player``, ``observation``, ``read_player_input`` and ``step``.

    A note on observations: there is a reserved string: "~~~END~~~" (in its own
    line), that is used to signal the end of the game. Environments must not
    use this in observations.
    """

    def __init__(self, num_players: int):
        self._num_players = num_players

    def reset(self, player_names: Optional[list[str]] = None) -> str:
        raise NotImplementedError()

    def next_player(self, current_player: Optional[int]) -> Optional[int]:
        """
        Calculate the index of the next player, return ``None`` if the game is
        over. ``current_player`` is ``None`` at the beginning, otherwise
        contains the current player index.
        """
        raise NotImplementedError()

    def observation(self, current_player: int) -> str:
        """
        Observation to be sent to the current player

        Can be a multiline string, in which case lines should be separarated by
        "\n"s. The final newline will be appended.
        """
        raise NotImplementedError()

    def read_player_input(
            self, read_line: Callable[[], str]) -> Optional[PlayerInput]:
        """
        Read and optionally parse/validate player input.

        Should take minimal time: player reply timeout is based on the timing
        of this function.

        Returns ``None`` on invalid player input.

        ``read_line`` returns one line of player input (without the ending
        newline, but check ``client_bridge.py`` to be sure).

        Not called if the player has been disqualified.
        """
        raise NotImplementedError()

    def invalid_player_input(self, current_player: int,
                             disqualified: bool) -> None:
        """
        Handle invalid player input.

        Default is to do nothing.

        Note that timeout also counts as invalid input.

        Parameter ``disqualified`` is indicates whether the player has been
        already disqualified (i.e., communication is broken off with them).
        """

    def step(self, current_player: int, player_input: PlayerInput) -> None:
        """
        Apply player action.
        """
        raise NotImplementedError()

    def get_scores(self) -> list[int | float]:
        """
        Return the scores of the players. Called after the end of the game.
        """
        raise NotImplementedError()

    @property
    def num_players(self):
        return self._num_players

class ClientInfo(NamedTuple):
    socket: socket.socket
    address_host: str
    address_port: int
    player_name: Optional[str] = None
    strikes: int = 0

    @property
    def disqualified(self):
        return self.strikes >= PLAYER_MAX_STRIKES

class PlaceholderClientInfo(NamedTuple):
    """
    A client that is not connected
    """
    player_name: Optional[str] = None
    disqualified: bool = True

class EnvironmentRunner:

    def __init__(self,
                 environment: EnvironmentBase,
                 step_timeout: float,
                 connection_timeout: float,
                 client_addresses: Optional[list[str]] = None,
                 player_names: Optional[list[str]] = None):
        self.env = environment
        self.step_timeout = step_timeout
        if client_addresses is not None:
            assert self.env.num_players == len(set(client_addresses)), (
                'Wrong number of clients for this environment or duplicate '
                'client addresses.')
        # Wait for players to connect
        server_socket = socket.create_server(('', network.JUDGE_PORT))
        server_socket.settimeout(connection_timeout)
        server_socket.listen(self.env.num_players)
        connected_clients: list[ClientInfo] = []
        print('Waiting for players to connect...')
        for _ in range(self.env.num_players):
            try:
                (clientsocket, address) = server_socket.accept()
                client_info = ClientInfo(clientsocket, address[0], address[1])
            except TimeoutError:
                print('Warning: connection timed out. May not have '
                      'enough players.')
                break
            clientsocket.settimeout(3 * self.step_timeout)
            address, _port = address
            if client_addresses and player_names:
                # find the name corresponding to the address
                player_name = player_names[client_addresses.index(
                    client_info.address_host)]
                client_info = client_info._replace(player_name=player_name)
            elif player_names:
                # find the name next in order
                player_name = player_names[len(connected_clients)]
                client_info = client_info._replace(player_name=player_name)
            else:
                player_name = str(len(connected_clients))
            connected_clients.append(client_info)
            print(f'Player {player_name} connected from {address}')
        if client_addresses is not None:
            addr_to_clients = {c.address_host: c for c in connected_clients}
            if len(addr_to_clients) != len(connected_clients):
                raise RuntimeError(
                    'Multiple connections from the same address: '
                    f'{[c.address_host for c in connected_clients]}')
            if not set(addr_to_clients.keys()).issubset(client_addresses):
                raise RuntimeError(
                    'Got invalid connections: '
                    f'{set(addr_to_clients.keys()) - set(client_addresses)}')
            clients: list[ClientInfo | PlaceholderClientInfo] = []
            for caddr in client_addresses:
                if caddr in addr_to_clients:
                    clients.append(addr_to_clients[caddr])
                else:
                    print(f'No connections from {caddr}.')
                    if player_names:
                        clients.append(
                            PlaceholderClientInfo(
                                player_names[client_addresses.index(caddr)]))
                    else:
                        clients.append(PlaceholderClientInfo())
        else:
            clients = connected_clients + [PlaceholderClientInfo()] * (
                self.env.num_players - len(connected_clients))
        self.clients = clients
        server_socket.close()
        self._client_reply_times: dict[int, list[float]] = {}

    def run(self) -> list[int | float]:
        print('Started the run.')
        self._send_initial_observations()
        current_player: Optional[int] = None
        while True:
            current_player = self.env.next_player(current_player)
            if current_player is None:
                break
            assert 0 <= current_player < self.env.num_players
            observation = self.env.observation(current_player)
            if not observation or observation[-1] != '\n':
                observation += '\n'
            self._send_observation(
                current_player, observation, only_qualified=True)
            send_tick = time.perf_counter()
            if self.clients[current_player].disqualified:
                player_input = None
            else:
                try:
                    tick = time.perf_counter()
                    player_input = self.env.read_player_input(
                        lambda: self._read_from_client(current_player))
                    tock = time.perf_counter()
                    if tock - tick > self.step_timeout:
                        player_input = None
                except TimeoutError:
                    player_input = None
                except network.NetworkError:
                    player_input = None
                tock = time.perf_counter()
                self._client_reply_times.setdefault(current_player,
                                                    []).append(tock - send_tick)
                if player_input is None:
                    cur_client = self.clients[current_player]
                    assert isinstance(cur_client, ClientInfo)
                    cur_strikes = cur_client.strikes
                    self.clients[current_player] = (
                        cur_client._replace(strikes=cur_strikes + 1))
                    if cur_client.strikes == PLAYER_MAX_STRIKES:
                        print(f'Player {self._player_name(current_player)} is '
                              'disqualified.')
            if player_input is None:
                self.env.invalid_player_input(
                    current_player, self.clients[current_player].disqualified)
            else:
                self.env.step(current_player, player_input)
        self._signal_the_end()
        return self.env.get_scores()

    def _player_name(self, player_ind: int) -> str:
        player_name = self.clients[player_ind].player_name
        if player_name:
            return player_name
        else:
            return str(player_ind)

    def _send_initial_observations(self) -> None:
        player_names = [c.player_name for c in self.clients]
        # mypy cannot resolve the `all` check
        initial_obs = self.env.reset(
            player_names if all(player_names) else None)  # type: ignore
        if not initial_obs or initial_obs[-1] != '\n':
            initial_obs += '\n'
        print('Sending initial observation to all players.')
        for p in range(self.env.num_players):
            self._send_observation(p, initial_obs, only_qualified=True)

    def _signal_the_end(self) -> None:
        print('Run ends, sending the end signal to everyone...')
        for p in range(self.env.num_players):
            self._send_observation(p, '~~~END~~~\n', only_qualified=False)

    def _send_observation(self,
                          current_player: int,
                          observation: str,
                          *,
                          only_qualified: bool = False):
        cur_client = self.clients[current_player]
        if only_qualified and cur_client.disqualified:
            return
        if getattr(cur_client, 'socket', None) is None:
            # Not connected
            return
        try:
            # Check for `socket` is done above
            network.send_data(cur_client.socket, observation)  # type: ignore
        except (TimeoutError, network.NetworkError):
            print(
                f'Failed to send to player {self._player_name(current_player)}.'
            )

    def _read_from_client(self, player_ind: int) -> str:
        cur_client = self.clients[player_ind]
        if getattr(cur_client, 'socket', None) is None:
            raise network.NetworkError(
                f'Player {self._player_name(player_ind)} not connected.')
        # Check for `socket` is done above, mypy doesn't see it
        msg = network.recv_msg(cur_client.socket)  # type: ignore
        assert msg['type'] == 'data', 'Control messages aren\'t supported yet.'
        return msg['data']

    @property
    def client_reply_times(self) -> dict[int | str, list[float]]:
        # yapf: disable
        return {
            # We check for None, but mypy fails to see it.
            self.clients[k].player_name # type: ignore
            if self.clients[k].player_name
            else k: v
            for k, v in sorted(
                self._client_reply_times.items(), key=lambda x: x[0])
        }
        # yapf: enable

class App:
    """
    Class mainly for parsing arguments and writing results where it is expected
    """

    def __init__(self, environment_name: str):
        arguments = self._parse_args(environment_name)
        print(environment_name)
        self._replay_file_path = arguments.replay_file
        self._player_timeout = arguments.timeout
        self._connection_timeout = arguments.connection_timeout
        config_file_path = arguments.config_file
        self._output_file_path = arguments.output_file
        with open(config_file_path, 'r') as f:
            self._options = json.load(f)
        if 'num_players' in self._options:
            print('Warning: number of players specified in configuration file, '
                  'replacing it with command line argument value '
                  f'({self._options["num_players"]}->{arguments.num_players}).')
        self._options['num_players'] = arguments.num_players
        if self._options['num_players'] <= 0:
            raise ValueError(
                f'Invalid number of players: {self._options["num_players"]}')
        if arguments.client_addresses:
            self._client_addresses = arguments.client_addresses.split(';')
            assert (len(self._client_addresses)
                    == self._options['num_players']), \
                    'Number of client addresses must equal the number of ' \
                    'players.'
        else:
            self._client_addresses = None
        if arguments.player_names:
            self._player_names = arguments.player_names.split(';')
            assert (len(self._player_names)
                    == self._options['num_players']), \
                    'Number of player names must equal the number of ' \
                    'players.'
        else:
            self._player_names = None

    @staticmethod
    def _parse_args(environment_name: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f'Judge program of the {environment_name} environment.')
        parser.add_argument(
            'config_file',
            type=str,
            default=None,
            help='Path to the environment config file.')
        parser.add_argument(
            'num_players',
            type=int,
            help='Number of players. Note that there is a maximum number for '
            'the different maps.')
        parser.add_argument(
            '--replay_file',
            type=str,
            default=None,
            help='Path to save replay file to. Optional, if omitted, no replay '
            'file is created.')
        parser.add_argument(
            '--output_file',
            type=str,
            help='Path to save the output file to. Optional.')
        parser.add_argument(
            '--timeout',
            type=float,
            default=1.,
            help='Timeout (in seconds) for the player responses. '
            'Default is 1.0 second.')
        parser.add_argument(
            '--connection_timeout',
            type=float,
            default=10,
            help='Timeout (in seconds) for player connections. '
            'Default is 10 second.')
        parser.add_argument(
            '--client_addresses',
            type=str,
            help='List of client addresses, separated by ";"s. The number '
            'of addresses must equal the number of players.')
        parser.add_argument(
            '--player_names',
            type=str,
            help='List of player names, separated by ";"s. The number '
            'of names must equal the number of players.')
        return parser.parse_args()

    @contextlib.contextmanager
    def replay_file(self):
        assert self._replay_file_path, 'No replay file path specified.'
        print(f'Saving replays to {self._replay_file_path}.')
        with open(self._replay_file_path, 'w') as f:
            yield f

    @property
    def create_replay(self):
        return bool(self._replay_file_path)

    def write_output(self, output):
        if self._output_file_path:
            print(f'Saving final scores to {self._output_file_path}.')
            with open(self._output_file_path, 'w') as f:
                json.dump(output, f)

    def run_environment(self,
                        env: EnvironmentBase,
                        *,
                        print_replay_times: bool | Literal['full'] = False):
        """
        Run a match in the environment.

        Arguments
        ---------
        env: EnvironmentBase
        print_replay_times: bool or 'full'
            If ``True``, prints mean of reply times (in seconds) for each
            agent. If "full", prints the full list of reply times for each
            agent.
        """
        runner = EnvironmentRunner(env, self._player_timeout,
                                   self._connection_timeout,
                                   self._client_addresses, self._player_names)
        scores = runner.run()
        if print_replay_times:
            avg_replay_times = {
                k: np.mean(v) if print_replay_times != 'full' else v
                for k, v in runner.client_reply_times.items()
            }
            print('Client reply times:')
            pprint(avg_replay_times, sort_dicts=False)
        return scores

    @property
    def options(self):
        return self._options

    @property
    def player_timeout(self):
        return self._player_timeout
