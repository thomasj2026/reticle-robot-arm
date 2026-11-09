from __future__ import annotations

import numpy as np


def pack(obj) -> bytes:
    import msgpack

    def enc(o):
        if isinstance(o, np.ndarray):
            return {b"__ndarray__": True, b"data": o.tobytes(), b"dtype": o.dtype.str, b"shape": o.shape}
        if isinstance(o, np.generic):
            return {b"__npgeneric__": True, b"data": o.item(), b"dtype": o.dtype.str}
        return o

    return msgpack.packb(obj, default=enc)


def unpack(buf: bytes):
    import msgpack

    def dec(o):
        if b"__ndarray__" in o:
            return np.ndarray(buffer=o[b"data"], dtype=np.dtype(o[b"dtype"]), shape=o[b"shape"])
        if b"__npgeneric__" in o:
            return np.dtype(o[b"dtype"]).type(o[b"data"])
        return o

    return msgpack.unpackb(buf, object_hook=dec)
