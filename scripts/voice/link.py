# Standard library only: the robot program imports CommandReceiver from its own environment.
import argparse
import json
import queue
import socket
import threading
import time

HOST = "127.0.0.1"
PORT = 8765
TIMEOUT_SECONDS = 1.0


def send(command: dict, host: str = HOST, port: int = PORT) -> bool:
    # one connection per command, so start order and restarts of either program do not matter
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT_SECONDS) as connection:
            connection.sendall(json.dumps(command).encode() + b"\n")
        return True
    except OSError:
        return False


class CommandReceiver:
    def __init__(self, host: str = HOST, port: int = PORT):
        self._commands: queue.Queue[dict] = queue.Queue()
        self._server = socket.create_server((host, port))
        threading.Thread(target=self._serve, daemon=True).start()

    def poll(self) -> dict | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._server.close()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            with connection:
                for line in receive_lines(connection):
                    command = decode(line)
                    if command is not None:
                        self._commands.put(command)


def receive_lines(connection: socket.socket) -> list[str]:
    connection.settimeout(TIMEOUT_SECONDS)
    data = b""
    try:
        while chunk := connection.recv(4096):
            data += chunk
    except OSError:
        pass
    return data.decode(errors="replace").splitlines()


def decode(line: str) -> dict | None:
    try:
        command = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(command, dict) and ({"target", "destination"} <= command.keys() or command.get("abort") is True
                                      or command.get("dance") is True or isinstance(command.get("plan"), list)):
        return command
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stand-in for the robot program: prints every command it receives.")
    parser.add_argument("--host", default=HOST, help="0.0.0.0 accepts commands from another machine")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    receiver = CommandReceiver(args.host, args.port)
    print(f"waiting for commands on {args.host}:{args.port}, Ctrl+C to stop", flush=True)
    try:
        while True:
            command = receiver.poll()
            if command is None:
                time.sleep(1 / 30)  # polled once per tick, like the robot program's 30 Hz control loop
            else:
                print(json.dumps(command), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
