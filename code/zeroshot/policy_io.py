# Minimal localhost protocol so policies can live in their own Python envs:
# the LIBERO client sends a pickled observation dict, the policy server replies with a 7-d LIBERO action.
import pickle, socket, struct
import numpy as np

def _send(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!Q", len(data)) + data)

def _recv(sock):
    n = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    return pickle.loads(_recv_exact(sock, n))

def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("policy connection closed")
        buf += chunk
    return bytes(buf)

class RemotePolicy:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port))
    def reset(self, instruction):
        _send(self.sock, {"cmd": "reset", "instruction": instruction}); return _recv(self.sock)
    def act(self, obs):
        _send(self.sock, {"cmd": "act", "obs": obs}); return np.asarray(_recv(self.sock))

def serve(policy, port):
    """policy needs .reset(instruction) and .act(obs) -> np.ndarray(7) in LIBERO convention."""
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(1)
    print(f"policy server ready on {port}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                msg = _recv(conn)
                if msg["cmd"] == "reset":
                    _send(conn, policy.reset(msg["instruction"]))
                else:
                    _send(conn, [float(x) for x in policy.act(msg["obs"])])   # plain floats: envs may run numpy 1 vs 2
        except ConnectionError:
            conn.close()
