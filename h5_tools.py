"""
h5_tools.py
-----------
Read/write Python dicts to/from HDF5 files using h5py.

Write rules (dict_to_hdf5)
--------------------------
- dict  → HDF5 group
- list / tuple of numbers → HDF5 dataset
- scalar (int, float, str, bool, bytes) → HDF5 scalar dataset
- numpy array → HDF5 dataset (gzip-compressed)
- None → stored as string sentinel "__None__"

Read (hdf5_to_dict)
-------------------
Works on any HDF5 file, not just files written by dict_to_hdf5:
- HDF5 group          → nested dict
- numeric dataset     → numpy array or Python scalar
- string dataset      → str or list[str]
- compound dataset    → list[dict] (1-D) or structured numpy array
- null/empty dataset  → None or empty numpy array
- group attributes    → included under "__attrs__" when load_attrs=True
"""

import h5py
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Core recursive writer
# --------------------------------------------------------------------------- #

def _write_header_attr(h5_group: h5py.Group, value) -> None:
    """Store a 'Header' value as HDF5 attributes on *h5_group*.

    - dict  → each key stored as a separate attribute
    - other → stored as h5_group.attrs["Header"]
    """
    if isinstance(value, dict):
        # Marker so the reader can reconstruct the "Header" dict on round-trip
        h5_group.attrs["__header_keys__"] = list(value.keys())
        for k, v in value.items():
            if isinstance(v, (list, tuple)):
                h5_group.attrs[str(k)] = np.array(v)
            elif isinstance(v, np.ndarray):
                h5_group.attrs[str(k)] = v
            elif v is None:
                h5_group.attrs[str(k)] = "__None__"
            else:
                h5_group.attrs[str(k)] = v
    elif isinstance(value, (list, tuple)):
        h5_group.attrs["Header"] = np.array(value)
    elif value is None:
        h5_group.attrs["Header"] = "__None__"
    else:
        h5_group.attrs["Header"] = value


def _write_node(h5_group: h5py.Group, key: str, value, overwrite: bool = False) -> None:
    """Recursively write *value* into *h5_group* under *key*."""

    if key == "Header":
        _write_header_attr(h5_group, value)
        return

    if isinstance(value, dict):
        sub = h5_group.require_group(key)
        for k, v in value.items():
            _write_node(sub, str(k), v, overwrite)
        return

    # Dataset path: remove existing key when overwrite is requested
    if overwrite and key in h5_group:
        del h5_group[key]

    if value is None:
        h5_group.create_dataset(key, data="__None__")

    elif isinstance(value, bool):
        # bool must come before int (bool is a subclass of int)
        ds = h5_group.create_dataset(key, data=int(value))
        ds.attrs["dtype_hint"] = "bool"

    elif isinstance(value, (int, float, complex)):
        h5_group.create_dataset(key, data=value)

    elif isinstance(value, str):
        # Use special_dtype for variable-length UTF-8 strings
        dt = h5py.string_dtype(encoding="utf-8")
        h5_group.create_dataset(key, data=value, dtype=dt)

    elif isinstance(value, bytes):
        h5_group.create_dataset(key, data=np.void(value))

    elif isinstance(value, np.ndarray):
        h5_group.create_dataset(key, data=value, compression="gzip")

    elif isinstance(value, (list, tuple)):
        arr = np.array(value)
        if arr.dtype.kind in ("U", "S", "O"):   # string / object array
            dt = h5py.string_dtype(encoding="utf-8")
            h5_group.create_dataset(key, data=arr.astype(object), dtype=dt)
        else:
            h5_group.create_dataset(key, data=arr, compression="gzip")

    else:
        raise TypeError(
            f"Unsupported type for key '{key}': {type(value).__name__}"
        )


def dict_to_hdf5(data: dict, path: str | Path, mode: str = "w",
                 overwrite: bool = False) -> None:
    """
    Write *data* (a potentially nested dict) to an HDF5 file at *path*.

    Parameters
    ----------
    data : dict
        The (nested) dictionary to serialise.
    path : str | Path
        Destination .h5 / .hdf5 file.
    mode : str
        File open mode — "w" (overwrite whole file) or "a" (append/update).
    overwrite : bool
        When True and mode="a", existing datasets are deleted and re-written
        instead of raising an error.  Has no effect when mode="w".
    """
    with h5py.File(path, mode) as f:
        for key, value in data.items():
            _write_node(f, str(key), value, overwrite)
    print(f"Wrote HDF5 → {path}")


# --------------------------------------------------------------------------- #
#  Core recursive reader
# --------------------------------------------------------------------------- #

def _read_attrs(h5_attrs) -> dict:
    """Convert HDF5 attributes to a plain Python dict."""
    out = {}
    for k, v in h5_attrs.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("S", "O"):
                dec = np.vectorize(
                    lambda x: x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
                )(v)
                out[k] = dec.item() if v.ndim == 0 else dec.tolist()
            elif v.ndim == 0:
                out[k] = v.item()
            else:
                out[k] = v.tolist()
        elif isinstance(v, (bytes, np.bytes_)):
            out[k] = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _read_node(node, load_attrs: bool = False):
    """Recursively read an HDF5 group or dataset into Python objects."""

    if isinstance(node, h5py.Group):
        result = {}
        for k in node:
            try:
                result[k] = _read_node(node[k], load_attrs)
            except Exception:
                result[k] = None  # dangling soft/external link or unreadable node

        if node.attrs:
            attrs = _read_attrs(node.attrs)

            # Reconstruct "Header" — dict case (written with __header_keys__ marker)
            if "__header_keys__" in attrs:
                keys = attrs.pop("__header_keys__")
                if isinstance(keys, str):
                    keys = [keys]
                header = {}
                for k in keys:
                    v = attrs.pop(k, None)
                    header[k] = None if v == "__None__" else v
                result["Header"] = header
            # Reconstruct "Header" — scalar/array case
            elif "Header" in attrs:
                v = attrs.pop("Header")
                result["Header"] = None if v == "__None__" else v

            if load_attrs and attrs:
                result["__attrs__"] = attrs

        return result

    # ---- Dataset ----
    if node.shape is None:  # null dataspace
        return None

    hint = node.attrs.get("dtype_hint", "")

    # String dtype (variable-length UTF-8 or fixed-length bytes)
    if h5py.check_string_dtype(node.dtype):
        raw = node[()]

        def _dec(x):
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return str(x) if not isinstance(x, (str, np.str_)) else x

        if node.ndim == 0:
            s = _dec(raw)
            return None if s == "__None__" else s
        decoded = np.vectorize(_dec)(np.asarray(raw, dtype=object))
        return decoded.tolist()

    # Compound (structured) dtype → list of dicts for 1-D arrays
    if node.dtype.names:
        value = node[()]
        if value.ndim == 1:
            rows = []
            for row in value:
                d = {}
                for name in node.dtype.names:
                    cell = row[name]
                    if isinstance(cell, bytes):
                        d[name] = cell.decode("utf-8", errors="replace")
                    elif isinstance(cell, np.generic):
                        d[name] = cell.item()
                    else:
                        d[name] = cell
                rows.append(d)
            return rows
        return value

    value = node[()]

    if hint == "bool":          # backward-compat with dict_to_hdf5
        return bool(value)
    if isinstance(value, np.void):  # raw bytes stored via np.void
        return bytes(value)
    if isinstance(value, np.generic):  # numpy scalar → Python scalar
        return value.item()
    return value  # ndarray (possibly empty) or other


def hdf5_to_dict(path: str | Path, load_attrs: bool = False) -> dict:
    """
    Read an arbitrary HDF5 file into a nested Python dict.

    Parameters
    ----------
    path : str | Path
        Path to the .h5 / .hdf5 file.
    load_attrs : bool
        If True, HDF5 group attributes are included under the key
        ``"__attrs__"`` in each corresponding dict level.  Default False.

    Returns
    -------
    dict
        Nested dict mirroring the HDF5 group/dataset hierarchy.
    """
    with h5py.File(path, "r") as f:
        result = {k: _read_node(v, load_attrs) for k, v in f.items()}
        if load_attrs:
            attrs = _read_attrs(f.attrs)
            if attrs:
                result["__attrs__"] = attrs
    return result




def merge_dicts(dict1, dict2):
    merged_dict = {}

    # header
    merged_dict["Header"] = {}
    for k in ["NumPart_ThisFile", "NumPart_Total"]:
        merged_dict["Header"][k] = [a + b for a, b in zip(dict1["Header"][k], dict2["Header"][k])]
    merged_dict["Header"]["Time"] = 0.0

    # particle data
    for i in range(6):
        ptype = "PartType%d" % i
        if (ptype in dict1) or (ptype in dict2):
            if (ptype in dict1) and (ptype in dict2):
                shared_keys = set(dict1[ptype]) & set(dict2[ptype])
                id_offset = len(dict1[ptype]["ParticleIDs"]) + 1 if "ParticleIDs" in shared_keys else 0
                merged_dict[ptype] = {}
                for k in shared_keys:
                    arr1 = dict1[ptype][k]
                    arr2 = dict2[ptype][k]
                    if k == "ParticleIDs":
                        arr2 = arr2 + id_offset
                    merged_dict[ptype][k] = np.concatenate([arr1, arr2], axis=0)
            elif ptype in dict1:
                merged_dict[ptype] = dict(dict1[ptype])
            else:
                merged_dict[ptype] = dict(dict2[ptype])

    return merged_dict



def merge_snapshots(filename1, filename2, output_filename):
    dict1 = snapshot_to_dict(filename1)
    dict2 = snapshot_to_dict(filename2)
    merged = merge_dicts(dict1, dict2)
    dict_to_snapshot(merged, output_filename)


def snapshot_to_dict(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as F:
        if "Header" in F:
            data["Header"] = dict(F["Header"].attrs)
        for i in range(6):
            ptype = "PartType%d" % i
            if ptype in F:
                data[ptype] = {k: F[ptype][k][:] for k in F[ptype]}
    return data


def dict_to_snapshot(data, filename):
    import h5py
    with h5py.File(filename, "w") as F:
        grp = F.create_group("Header")
        for k in data["Header"]:
            grp.attrs[k] = data["Header"][k]
        for i in range(6):
            ptype = "PartType%d" % i
            if ptype in data:
                grp = F.create_group(ptype)
                for k in data[ptype]:
                    grp.create_dataset(k, data=data[ptype][k])



# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sample = {
        "metadata": {
            "experiment": "alpha",
            "version": 3,
            "tags": ["fast", "reliable"],
            "active": True,
        },
        "config": {
            "learning_rate": 1e-3,
            "layers": [64, 128, 64],
            "dropout": 0.25,
            "notes": None,
        },
        "weights": {
            "layer_0": np.random.randn(64, 32).astype(np.float32),
            "layer_1": np.random.randn(128, 64).astype(np.float32),
        },
        "scalar_int": 42,
        "scalar_float": 3.14159,
        "top_string": "hello HDF5",
    }

    out_path = Path("/tmp/sample.h5")
    dict_to_hdf5(sample, out_path)

    # ---- round-trip verify ----
    recovered = hdf5_to_dict(out_path)
    print("\nRound-trip check:")
    print(f"  metadata.experiment : {recovered['metadata']['experiment']}")
    print(f"  metadata.active     : {recovered['metadata']['active']!r}")
    print(f"  config.notes        : {recovered['config']['notes']!r}")
    print(f"  scalar_int          : {recovered['scalar_int']}")
    print(f"  weights.layer_0     : shape {recovered['weights']['layer_0'].shape}")

    # ---- show HDF5 structure with h5py ----
    print("\nHDF5 structure:")
    with h5py.File(out_path, "r") as f:
        f.visititems(lambda name, obj: print(
            f"  {'GROUP' if isinstance(obj, h5py.Group) else 'DSET ':5s}  /{name}"
        ))