#!/usr/bin/env python
import os
import subprocess
import asyncio
import datetime
import threading
import socket
import argparse
import sys
import textwrap
from typing import Optional
import network

LOGGING = True
BOT_READY_SIGNAL = 'READY'
#advanced loggings
LOG_DIR = os.path.join(os.getcwd(), "logs")
#advanced loggings

class Logger:

    def __init__(self, fname: str):
        self.f = open(fname, 'w')  # pylint: disable=consider-using-with
        self.lock = threading.Lock()

    def write_stdout(self, msg: str):
        with self.lock:
            self.f.write(
                f'{datetime.datetime.now().time().isoformat()} - stdout  :: '
                f'{msg}\n')
            self.f.flush()

    def write_stderr(self, msg: str):
        with self.lock:
            self.f.write(
                f'{datetime.datetime.now().time().isoformat()} - stderr  :: '
                f'{msg}\n')
            self.f.flush()

    def write_stdin(self, msg: str):
        with self.lock:
            self.f.write(
                f'{datetime.datetime.now().time().isoformat()} - stdin   :: '
                f'{msg}\n')
            self.f.flush()

    def write_control(self, msg: str):
        with self.lock:
            self.f.write(
                f'{datetime.datetime.now().time().isoformat()} - control :: '
                f'{msg}\n')
            self.f.flush()

    def close(self):
        with self.lock:
            self.f.close()

class SubmissionManager():
    socket: socket.socket
    # Pylint doesn't find `Process`
    submission_process: asyncio.subprocess.Process  # pylint: disable=no-member
    logger: Optional[Logger]

    def __init__(self, judge_address: str, exe_cmd: list[str],
                 init_timeout: float) -> None:
        if LOGGING:
            #advanced loggings
            os.makedirs(LOG_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
            log_path = os.path.join(LOG_DIR, f"communication.{ts}.log")
            self.logger = Logger(log_path)
            #advanced loggings
        else:
            self.logger = None
        self._judge_address = judge_address
        self._exe_cmd = exe_cmd
        self._init_timeout = init_timeout

    async def start(self):
        # Start submitted program
        if self.logger is not None:
            self.logger.write_control('Starting bot process.')
        self.submission_process = await asyncio.create_subprocess_exec(
            *self._exe_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        try:
            await self.bot_initialisation()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.read_stdout())
                tg.create_task(self.listen_to_server())
                if self.logger is not None:
                    tg.create_task(self.read_stderr())
        except RuntimeError as e:
            if self.logger is not None:
                self.logger.write_control(f'Exiting: {e}')
            print('Exiting:', e)
        finally:
            await self.close()

    async def bot_initialisation(self) -> None:
        """
        Wait until the bot initialises then connect to server
        """
        assert self.submission_process.stdout is not None
        try:
            line: str = (await asyncio.wait_for(
                self.submission_process.stdout.readline(),
                timeout=self._init_timeout)).decode('utf8')
        except TimeoutError as e:
            raise RuntimeError('Bot initialisation timeout') from e
        if not line:
            raise RuntimeError('Bot did not initialise.')
        line = line.removesuffix('\n')  # see ``read_stdout``
        if line != BOT_READY_SIGNAL:
            print(f'Warning: first line from bot is not {BOT_READY_SIGNAL}:\n'
                  f'{textwrap.shorten(line, 80)}')
        if self.logger is not None:
            self.logger.write_control(
                'Bot has initialised, connecting to server.')
        # Connect to judge
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self._judge_address, network.JUDGE_PORT))

    async def read_stdout(self):
        assert self.submission_process.stdout is not None
        try:
            while True:
                # ``readline`` will return the ending newline, this is good
                # when the line is empty (i.e., it will return a string with
                # the newline character, and it will not quit the loop). At
                # EOF, however, it will return an empty string.
                line: str = (
                    await
                    self.submission_process.stdout.readline()).decode('utf8')
                if not line:
                    break
                line = line.removesuffix('\n')
                if self.logger is not None:
                    self.logger.write_stdout(line)
                network.send_data(self.socket, line)
        except network.NetworkError:
            if self.logger is not None:
                self.logger.write_control(
                    f'Failed to send last line to server:\n{line}')

    async def read_stderr(self):
        # stderr goes only to logging, this thread shouldn't have been
        # started otherwise
        assert LOGGING
        assert self.logger is not None
        assert self.submission_process.stderr is not None
        while True:
            line: str = (
                await self.submission_process.stderr.readline()).decode('utf8')
            if not line:
                break
            line = line.removesuffix('\n')
            self.logger.write_stderr(line)

    async def listen_to_server(self):
        assert self.submission_process.stdin is not None

        def wait_for_message():
            return network.recv_msg(self.socket)

        try:
            while True:
                msg = await asyncio.to_thread(wait_for_message)
                assert msg['type'] == 'data', \
                        f'{msg["type"]} messages aren\'t supported yet.'
                if self.logger is not None:
                    self.logger.write_stdin(msg['data'][:-1])
                self.submission_process.stdin.write(msg['data'].encode('utf8'))
                await self.submission_process.stdin.drain()
        except network.NetworkError:
            if self.logger is not None:
                self.logger.write_control(
                    'Server has terminated, no more data.')
        except ConnectionResetError:
            print('Error: can\'t write to client. Maybe it terminated?')

    async def close(self) -> None:
        submission_process = getattr(self, 'submission_process', None)
        if (submission_process is not None
                and submission_process.returncode is None):
            submission_process.terminate()
            await submission_process.wait()
        if self.logger is not None:
            self.logger.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        'A script that forwards bot standard input/output to the judge via '
        'the network.')
    parser.add_argument(
        'bot_exe',
        help='Path to the bot executable (must have executable or read '
        'permissions).')
    parser.add_argument(
        '--judge_address',
        type=str,
        default='localhost',
        help='Address of the judge system. Default is localhost.')
    parser.add_argument(
        '--init_timeout',
        type=float,
        default=5,
        help='Timeout (in seconds) for bot initialisation. Default is 5 '
        'seconds.')
    return parser.parse_args()

def get_execute_command(fname: str) -> list[str]:
    """
    Return the command to execute the bot
    """
    if fname.endswith('.py'):
        return ['python', '-u', fname]
    if fname.endswith('.mjs'):
        return ['node', fname]
    if fname.endswith('.lua'):
        return ['lua', fname]
    if os.path.splitext(fname)[1] == '':
        return [fname]
    print('Error: unknown filetype. Exiting.', file=sys.stderr)
    return []

def main():
    args = parse_args()
    cmd = get_execute_command(args.bot_exe)
    if not cmd:
        return
    try:
        manager = SubmissionManager(args.judge_address, cmd, args.init_timeout)
        asyncio.run(manager.start())
    except KeyboardInterrupt:
        manager.close()
        print('Received keyboard interrupt. Bye.')

if __name__ == "__main__":
    main()
