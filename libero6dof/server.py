from __future__ import annotations

import argparse
import importlib
import traceback

import numpy as np

from .wire import pack as _pack, unpack as _unpack


class _Hold:
    """Placeholder policy: returns a short chunk that holds still."""

    def act(self, obs):
        return np.tile(np.array([0, 0, 0, 0, 0, 0, -1.0], np.float32), (10, 1))


def load_policy(spec: str):
    if spec == "hold":
        return _Hold()
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)()


def serve(policy, host="0.0.0.0", port=8000, metadata=None):
    from websockets.sync.server import serve as ws_serve

    from websockets.exceptions import ConnectionClosed

    def handler(ws):
        ws.send(_pack(metadata or {}))
        try:
            _loop(ws)
        except ConnectionClosed:
            pass

    def _loop(ws):
        for msg in ws:
            try:
                obs = _unpack(msg)
                actions = np.asarray(policy.act(obs), dtype=np.float32)
                ws.send(_pack({"actions": actions}))
            except Exception:
                ws.send(traceback.format_exc())

    with ws_serve(handler, host, port, compression=None, max_size=None) as server:
        print(f"policy server listening on ws://{host}:{port}")
        server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="hold", help="'hold' or module.path:factory")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    serve(load_policy(a.policy), a.host, a.port)


if __name__ == "__main__":
    main()
