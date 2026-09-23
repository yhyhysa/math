"""Unified readers for the three attachment packaging conventions.

Packaging conventions (explicit structural dispatch, never hidden behind try/except):

  1. A2 `aligned_50.pkl` / `unaligned_50.pkl` (Columnar split-dict)
     {"train": {col: array[N,...]}, "valid": {...}, "test": {...}}
     Unpacks to N samples per split: train=3395, valid=728, test=727.
     Each sample has text(50,768), text_bert(3,50), audio(50,74), vision(50,35),
     classification_label, regression_label, id, raw_text.

  2. A3 `附件3_XX.pkl` (Aligned test-dict with leading batch=1)
     {"test": {"audio": (1,50,74), "text_bert": (1,3,50), "vision": (1,50,35)}}
     Unpacks to 1 sample in split 'test'. Leading batch dim is stripped.
     sample_id derived from filename stem (e.g. '附件3_01').
     Does NOT manufacture official ID or synthetic text vector.

  3. A4 `XX.pkl` (Flat single-sample dict without batch dim)
     {"id": "01", "raw_text": "...", "text": (50,768), "text_bert": (3,50), "audio": (50,74), "vision": (50,35)}
     Unpacks to 1 sample in split 'single'. Retains original ID, raw_text, all 4 modalities.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLIT_KEYS = ("train", "valid", "test")
MODALITY_KEYS = ("text", "text_bert", "audio", "vision")


@dataclass
class Sample:
    """One clip. Modality arrays are stored WITHOUT any batch dimension."""

    sample_id: str
    raw_text: str | None
    arrays: dict[str, np.ndarray]
    source_file: str
    packaging: str
    classification_label: float | None = None
    regression_label: float | None = None
    original_shapes: dict[str, tuple] = field(default_factory=dict)
    squeezed: dict[str, bool] = field(default_factory=dict)
    extra_keys: dict[str, object] = field(default_factory=dict)

    def has(self, key: str) -> bool:
        return key in self.arrays

    def get_array(self, key: str) -> np.ndarray | None:
        return self.arrays.get(key)


@dataclass
class Payload:
    """A whole pickle file with explicit packaging metadata."""

    path: str
    top_keys: list[str]
    packaging: str
    splits: dict[str, list[Sample]]
    notes: list[str] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return sum(len(s) for s in self.splits.values())


def _is_columnar_dict(d: dict) -> tuple[bool, int]:
    """Check if dictionary has columnar arrays of identical length N > 1."""
    if not isinstance(d, dict):
        return False, 0
    lengths: dict[str, int] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            lengths[k] = v.shape[0]
        elif isinstance(v, (list, tuple)):
            lengths[k] = len(v)
    if not lengths:
        return False, 0
    uniq = set(lengths.values())
    # If all columns have the same length and length > 1, it's columnar
    if len(uniq) == 1 and next(iter(uniq)) > 1:
        return True, next(iter(uniq))
    # If majority agree and max > 1
    max_len = max(lengths.values())
    if max_len > 1:
        return True, max_len
    return False, 0


def load_pickle(path: str | Path, strict: bool = True) -> Payload:
    """Load one pickle file with strict, explicit structural dispatch.

    Never uses broad try/except to silence structure errors.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle file not found: {p}")

    with open(p, "rb") as fh:
        obj = pickle.load(fh)

    if not isinstance(obj, dict):
        raise ValueError(f"{p.name}: top-level type is {type(obj).__name__}, expected dict")

    top_keys = list(obj.keys())
    notes: list[str] = []

    # -------------------------------------------------------------------------
    # Case 1: Dict with SPLIT_KEYS ('train', 'valid', 'test')
    # -------------------------------------------------------------------------
    split_like = [k for k in SPLIT_KEYS if k in top_keys]
    if split_like:
        splits: dict[str, list[Sample]] = {}

        for k in split_like:
            inner = obj[k]
            if not isinstance(inner, dict):
                raise ValueError(f"{p.name}: split '{k}' expected dict, got {type(inner).__name__}")

            is_col, n_samples = _is_columnar_dict(inner)

            if is_col and n_samples > 1:
                # Subcase 1A: Columnar split (Attachment 2: aligned_50.pkl)
                samples = []
                col_keys = list(inner.keys())
                ids_col = inner.get("id")
                raw_texts_col = inner.get("raw_text")
                c_labels_col = inner.get("classification_labels")
                r_labels_col = inner.get("regression_labels")

                for i in range(n_samples):
                    sid = str(ids_col[i]) if ids_col is not None and i < len(ids_col) else f"{p.stem}_{k}_{i}"
                    raw_txt = str(raw_texts_col[i]) if raw_texts_col is not None and i < len(raw_texts_col) else None
                    c_lbl = float(c_labels_col[i]) if c_labels_col is not None and i < len(c_labels_col) else None
                    r_lbl = float(r_labels_col[i]) if r_labels_col is not None and i < len(r_labels_col) else None

                    arrays: dict[str, np.ndarray] = {}
                    orig_shapes: dict[str, tuple] = {}
                    for mod_k in MODALITY_KEYS:
                        if mod_k in inner:
                            arr = inner[mod_k][i]
                            orig_shapes[mod_k] = tuple(arr.shape)
                            arrays[mod_k] = np.asarray(arr)

                    samples.append(Sample(
                        sample_id=sid,
                        raw_text=raw_txt,
                        arrays=arrays,
                        source_file=p.name,
                        packaging="split-columnar",
                        classification_label=c_lbl,
                        regression_label=r_lbl,
                        original_shapes=orig_shapes,
                        squeezed={mk: False for mk in arrays},
                    ))

                splits[k] = samples
                notes.append(f"split '{k}' unpacked as columnar with N={n_samples}")

            else:
                # Subcase 1B: Single-sample dict under split (Attachment 3: 附件3_XX.pkl)
                # Has a leading batch dimension of 1 in arrays
                arrays = {}
                orig_shapes = {}
                squeezed = {}

                for key, val in inner.items():
                    if isinstance(val, np.ndarray):
                        orig_shapes[key] = tuple(val.shape)
                        if val.ndim >= 2 and val.shape[0] == 1:
                            arrays[key] = np.asarray(val[0])
                            squeezed[key] = True
                        else:
                            arrays[key] = np.asarray(val)
                            squeezed[key] = False

                # Derive sample_id from filename stem to avoid fake official ID
                sid = str(inner.get("id")) if "id" in inner and inner["id"] is not None else p.stem
                raw_txt = str(inner.get("raw_text")) if "raw_text" in inner and inner["raw_text"] is not None else None

                sample = Sample(
                    sample_id=sid,
                    raw_text=raw_txt,
                    arrays=arrays,
                    source_file=p.name,
                    packaging="split-single-sample-batch1",
                    classification_label=None,
                    regression_label=None,
                    original_shapes=orig_shapes,
                    squeezed=squeezed,
                )
                splits[k] = [sample]
                notes.append(f"split '{k}' unpacked as single sample with batch=1 squeezed; sample_id='{sid}'")

        pkg_name = "split-columnar" if any("columnar" in n for n in notes) else "split-single-sample"
        return Payload(str(p), top_keys, pkg_name, splits, notes)

    # -------------------------------------------------------------------------
    # Case 2: Flat single-sample dict without split keys (Attachment 4: 01.pkl ~ 20.pkl)
    # -------------------------------------------------------------------------
    if any(k in top_keys for k in ("text", "text_bert", "audio", "vision", "raw_text")):
        arrays = {}
        orig_shapes = {}
        squeezed = {}
        extra = {}

        for key, val in obj.items():
            if isinstance(val, np.ndarray):
                orig_shapes[key] = tuple(val.shape)
                if val.ndim >= 3 and val.shape[0] == 1:
                    arrays[key] = np.asarray(val[0])
                    squeezed[key] = True
                else:
                    arrays[key] = np.asarray(val)
                    squeezed[key] = False
            elif key in ("id", "raw_text"):
                extra[key] = val

        sid = str(obj.get("id", p.stem))
        raw_txt = str(obj.get("raw_text", "")) if "raw_text" in obj else None

        sample = Sample(
            sample_id=sid,
            raw_text=raw_txt,
            arrays=arrays,
            source_file=p.name,
            packaging="flat-single-sample",
            classification_label=None,
            regression_label=None,
            original_shapes=orig_shapes,
            squeezed=squeezed,
            extra_keys=extra,
        )
        notes.append(f"flat single sample unpacked without split; sample_id='{sid}'")
        return Payload(str(p), top_keys, "flat-single-sample", {"single": [sample]}, notes)

    raise ValueError(
        f"{p.name}: unrecognized packaging structure with top-level keys {top_keys!r}. "
        f"Refusing to guess."
    )
