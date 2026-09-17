"""Wait for model loading without starting simulator episodes prematurely."""
import os
import sys
import time
import zmq
from gala.eval.service import MsgSerializer

port, pid, timeout = map(int, sys.argv[1:4])
deadline = time.monotonic() + timeout
with zmq.Context() as context:
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit('Inference process exited; inspect the server log.')
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 1000)
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            socket.connect(f'tcp://127.0.0.1:{port}')
            try:
                socket.send(MsgSerializer.to_bytes({'endpoint': 'ping'}))
                reply = MsgSerializer.from_bytes(socket.recv())
                if 'error' not in reply:
                    print(f'Inference server ready on port {port}')
                    break
            except zmq.Again:
                pass
        time.sleep(1)
    else:
        raise SystemExit(f'Inference startup timed out after {timeout}s; inspect the server log.')
