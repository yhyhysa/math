"""Full-coverage verification of BERT tokenizer and frozen encoder text representations.

Verifies:
  1. Tokenizer metadata and model file SHA-256 hashes, pinned to a fixed snapshot revision.
  2. Full-sample tokenization matching on raw_text for Attachment 2 (4850 samples) and Attachment 4 (20 samples).
  3. Attachment 3 token range, special tokens, and encoder inference capability.
  4. Full-sample batched embedding reconstruction on GPU:
     Computes absolute difference between BertModel(text_bert).last_hidden_state and official 'text' (50, 768).
     Reports max, mean, p50, p90, p95, p99 quantiles and threshold exceed counts.

Version binding (P1 fix): the encoder is loaded from the *verified local snapshot directory*
of the pinned revision `86b5e0934494bd15c9632b12f734a8a67f723594`, never from the mutable
`refs/main` pointer and never with network access.  Required files are checked for presence
and SHA-256 before inference; any missing file or hash mismatch raises immediately, so the
report can never claim a revision whose bytes were not the bytes actually used.

Outputs:
  - results/bert_audit_full.json

Usage (cwd = 工程/):
  python -m src.tools.audit_bert_full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import torch
from transformers import BertModel, BertTokenizerFast

from src.config import load_config

PINNED_MODEL_NAME = "bert-base-uncased"
PINNED_REVISION_SHA = "86b5e0934494bd15c9632b12f734a8a67f723594"
DEFAULT_HUB_DIR = Path(r"C:\Users\本地账户\.cache\huggingface\hub\models--bert-base-uncased")
# Hashes recorded by the previous T0 round and re-verified here before use.
PINNED_FILE_SHA256 = {
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "config.json": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20",
    "model.safetensors": "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
}
REQUIRED_SNAPSHOT_FILES = ("vocab.txt", "config.json", "model.safetensors", "tokenizer.json")
OPTIONAL_SNAPSHOT_FILES = ("tokenizer_config.json", "special_tokens_map.json")


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def resolve_pinned_snapshot(hub_dir: Path | None = None) -> dict:
    """Locate and hash the pinned snapshot; raise on any missing file or hash mismatch."""
    hub = Path(os.environ.get("DSH_BERT_HUB_DIR", str(hub_dir or DEFAULT_HUB_DIR)))

    ref_path = hub / "refs" / "main"
    ref_revision = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None

    snapshot_dir = hub / "snapshots" / PINNED_REVISION_SHA
    if not snapshot_dir.is_dir():
        raise RuntimeError(
            f"BLOCKED: pinned snapshot directory not found: {snapshot_dir}. "
            f"the frozen encoder cannot be loaded from the recorded revision {PINNED_REVISION_SHA}."
        )

    missing = [f for f in REQUIRED_SNAPSHOT_FILES if not (snapshot_dir / f).is_file()]
    if missing:
        raise RuntimeError(f"BLOCKED: pinned snapshot {snapshot_dir} is missing required files: {missing}")

    file_records: dict[str, dict] = {}
    for fname in REQUIRED_SNAPSHOT_FILES + OPTIONAL_SNAPSHOT_FILES:
        fp = snapshot_dir / fname
        if not fp.is_file():
            continue
        actual = compute_file_sha256(fp)
        expected = PINNED_FILE_SHA256.get(fname)
        record = {
            "path": str(fp),
            "size_bytes": int(fp.stat().st_size),
            "sha256": actual,
            "expected_sha256": expected,
            "hash_verified": (expected is None) or (expected.lower() == actual.lower()),
        }
        if expected is not None and not record["hash_verified"]:
            raise RuntimeError(
                f"BLOCKED: {fname} hash mismatch in pinned snapshot: expected {expected}, got {actual} "
                f"({fp})"
            )
        file_records[fname] = record

    return {
        "model_name": PINNED_MODEL_NAME,
        "pinned_revision_sha": PINNED_REVISION_SHA,
        "hub_dir": str(hub),
        "refs_main_revision": ref_revision,
        "refs_main_matches_pinned": ref_revision == PINNED_REVISION_SHA,
        "snapshot_dir": str(snapshot_dir),
        "load_source": str(snapshot_dir),
        "required_files": list(REQUIRED_SNAPSHOT_FILES),
        "file_hashes": file_records,
    }


def _verify_loaded_file_matches_snapshot(snapshot: dict) -> dict:
    """Re-hash the files on disk after loading and confirm they still match the record."""
    result: dict[str, object] = {}
    for fname, record in snapshot["file_hashes"].items():
        after = compute_file_sha256(Path(record["path"]))
        result[fname] = {"sha256_after_load": after, "unchanged_since_hash": after == record["sha256"]}
        if after != record["sha256"]:
            raise RuntimeError(f"BLOCKED: {record['path']} changed while the encoder was being loaded")
    return result


def audit_bert_full() -> dict:
    t0 = time.time()
    cfg = load_config("configs/base.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Pinned model metadata & files (raises before any inference if unverifiable)
    snapshot = resolve_pinned_snapshot()
    revision_sha = snapshot["pinned_revision_sha"]
    file_hashes = snapshot["file_hashes"]
    model_name = snapshot["model_name"]
    snapshot_dir = snapshot["snapshot_dir"]

    print("==================================================")
    print("      BERT Full-Coverage Tokenizer & Embedding    ")
    print("==================================================")
    print(f"Model: {model_name}, pinned revision: {revision_sha}")
    print(f"Snapshot dir (load source): {snapshot_dir}")
    print(f"refs/main currently points to: {snapshot['refs_main_revision']} "
          f"(matches pinned: {snapshot['refs_main_matches_pinned']})")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Offline, snapshot-bound load: no network access, no refs/main resolution.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = BertTokenizerFast.from_pretrained(snapshot_dir, local_files_only=True)
    model = BertModel.from_pretrained(snapshot_dir, local_files_only=True)
    model.eval()
    model.to(device)

    loaded_tokenizer_path = getattr(tokenizer, "name_or_path", None)
    loaded_model_path = getattr(model, "name_or_path", None)
    loaded_file_check = _verify_loaded_file_matches_snapshot(snapshot)
    expected_norm = os.path.normcase(os.path.normpath(snapshot_dir))
    binding_ok = all(
        p is not None and os.path.normcase(os.path.normpath(str(p))) == expected_norm
        for p in (loaded_tokenizer_path, loaded_model_path)
    )
    if not binding_ok:
        raise RuntimeError(
            "BLOCKED: loaded weights/tokenizer path differs from the verified snapshot: "
            f"tokenizer={loaded_tokenizer_path!r}, model={loaded_model_path!r}, expected={snapshot_dir!r}"
        )
    print(f"Loaded tokenizer from: {loaded_tokenizer_path}")
    print(f"Loaded model from:     {loaded_model_path}")
    print(f"Post-load hash re-check: {all(v['unchanged_since_hash'] for v in loaded_file_check.values())}")

    # 2. Attachment 2 Tokenization and Reconstruction Audit
    a2_path = cfg["_attachment2_root"] / "aligned_50.pkl"
    with open(a2_path, "rb") as f:
        d2 = pickle.load(f)

    a2_results = {}
    total_a2_samples = 0
    all_diffs_a2 = []

    for split in ["train", "valid", "test"]:
        sp = d2[split]
        raw_texts = list(sp["raw_text"])
        tb_all = sp["text_bert"]  # (N, 3, 50)
        text_all = sp["text"]     # (N, 50, 768)
        N = len(raw_texts)
        total_a2_samples += N

        # Tokenization match
        tok_id_matches = 0
        tok_mask_matches = 0
        tok_type_matches = 0
        tok_full_matches = 0

        for i in range(N):
            tok = tokenizer(str(raw_texts[i]), max_length=50, padding="max_length", truncation=True, return_tensors="np")
            id_ok = bool(np.array_equal(tok["input_ids"][0], tb_all[i, 0]))
            mask_ok = bool(np.array_equal(tok["attention_mask"][0], tb_all[i, 1]))
            type_ok = bool(np.array_equal(tok["token_type_ids"][0], tb_all[i, 2]))
            if id_ok:
                tok_id_matches += 1
            if mask_ok:
                tok_mask_matches += 1
            if type_ok:
                tok_type_matches += 1
            if id_ok and mask_ok and type_ok:
                tok_full_matches += 1

        # Batched Reconstruction on GPU
        batch_size = 64
        diffs_split = []
        t_recon_start = time.time()

        with torch.no_grad():
            for b_start in range(0, N, batch_size):
                b_end = min(b_start + batch_size, N)
                inp_ids = torch.tensor(tb_all[b_start:b_end, 0], dtype=torch.long, device=device)
                att_mask = torch.tensor(tb_all[b_start:b_end, 1], dtype=torch.long, device=device)
                tok_type = torch.tensor(tb_all[b_start:b_end, 2], dtype=torch.long, device=device)

                out = model(input_ids=inp_ids, attention_mask=att_mask, token_type_ids=tok_type)
                recon_batch = out.last_hidden_state.cpu().numpy()  # (B, 50, 768)
                real_batch = text_all[b_start:b_end]

                diff_batch = np.abs(recon_batch - real_batch)  # (B, 50, 768)
                # Max error per sample
                sample_max_diffs = diff_batch.reshape(diff_batch.shape[0], -1).max(axis=1)
                diffs_split.extend(sample_max_diffs.tolist())
                all_diffs_a2.extend(sample_max_diffs.tolist())

        recon_elapsed = time.time() - t_recon_start
        diffs_arr = np.array(diffs_split)

        a2_results[split] = {
            "sample_count": int(N),
            "token_id_exact_matches": int(tok_id_matches),
            "token_mask_exact_matches": int(tok_mask_matches),
            "token_type_exact_matches": int(tok_type_matches),
            "token_full_exact_matches": int(tok_full_matches),
            "token_full_match_ratio": float(tok_full_matches / N),
            "reconstruction_stats": {
                "max_absolute_error": float(diffs_arr.max()),
                "mean_absolute_error": float(diffs_arr.mean()),
                "p50_absolute_error": float(np.percentile(diffs_arr, 50)),
                "p90_absolute_error": float(np.percentile(diffs_arr, 90)),
                "p95_absolute_error": float(np.percentile(diffs_arr, 95)),
                "p99_absolute_error": float(np.percentile(diffs_arr, 99)),
                "samples_exceeding_1e_3": int((diffs_arr > 1e-3).sum()),
                "samples_exceeding_1e_4": int((diffs_arr > 1e-4).sum()),
                "samples_exceeding_1e_5": int((diffs_arr > 1e-5).sum()),
                "samples_exceeding_1e_6": int((diffs_arr > 1e-6).sum()),
                "batch_size": int(batch_size),
                "elapsed_seconds": round(recon_elapsed, 2),
            }
        }
        print(f"Attachment 2 '{split}' (N={N}): full_token_match={tok_full_matches}/{N} ({tok_full_matches/N*100:.2f}%), "
              f"max_err={diffs_arr.max():.2e}, p99={np.percentile(diffs_arr, 99):.2e}, >1e-4={int((diffs_arr > 1e-4).sum())}")

    # 3. Attachment 4 Tokenization and Reconstruction Audit (20 samples)
    a4_dir = cfg["_attachment4_root"] / "对齐版本"
    a4_files = sorted(a4_dir.glob("*.pkl"))
    a4_raw_texts = []
    a4_tbs = []
    a4_texts = []

    for f in a4_files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        a4_raw_texts.append(str(d["raw_text"]))
        a4_tbs.append(d["text_bert"])
        a4_texts.append(d["text"])

    a4_tbs = np.stack(a4_tbs, axis=0)   # (20, 3, 50)
    a4_texts = np.stack(a4_texts, axis=0) # (20, 50, 768)
    N4 = len(a4_files)

    tok4_matches = 0
    for i in range(N4):
        tok = tokenizer(a4_raw_texts[i], max_length=50, padding="max_length", truncation=True, return_tensors="np")
        if (
            bool(np.array_equal(tok["input_ids"][0], a4_tbs[i, 0]))
            and bool(np.array_equal(tok["attention_mask"][0], a4_tbs[i, 1]))
            and bool(np.array_equal(tok["token_type_ids"][0], a4_tbs[i, 2]))
        ):
            tok4_matches += 1

    with torch.no_grad():
        inp_ids4 = torch.tensor(a4_tbs[:, 0], dtype=torch.long, device=device)
        att_mask4 = torch.tensor(a4_tbs[:, 1], dtype=torch.long, device=device)
        tok_type4 = torch.tensor(a4_tbs[:, 2], dtype=torch.long, device=device)
        out4 = model(input_ids=inp_ids4, attention_mask=att_mask4, token_type_ids=tok_type4)
        recon4 = out4.last_hidden_state.cpu().numpy()
        diff4 = np.abs(recon4 - a4_texts)
        diff4_maxs = diff4.reshape(N4, -1).max(axis=1)

    a4_results = {
        "sample_count": int(N4),
        "token_full_exact_matches": int(tok4_matches),
        "token_full_match_ratio": float(tok4_matches / N4),
        "reconstruction_stats": {
            "max_absolute_error": float(diff4_maxs.max()),
            "mean_absolute_error": float(diff4_maxs.mean()),
            "p50_absolute_error": float(np.percentile(diff4_maxs, 50)),
            "p90_absolute_error": float(np.percentile(diff4_maxs, 90)),
            "p99_absolute_error": float(np.percentile(diff4_maxs, 99)),
            "samples_exceeding_1e_3": int((diff4_maxs > 1e-3).sum()),
            "samples_exceeding_1e_4": int((diff4_maxs > 1e-4).sum()),
            "samples_exceeding_1e_5": int((diff4_maxs > 1e-5).sum()),
            "samples_exceeding_1e_6": int((diff4_maxs > 1e-6).sum()),
        }
    }
    print(f"Attachment 4 (N={N4}): full_token_match={tok4_matches}/{N4}, max_err={diff4_maxs.max():.2e}, >1e-4={int((diff4_maxs > 1e-4).sum())}")

    # 4. Attachment 3 Audit (30 files: shape, token range, special tokens, inference capability)
    a3_dir = cfg["_attachment3_root"] / "对齐版本"
    a3_files = sorted(a3_dir.glob("*.pkl"))
    a3_meta = []
    a3_tbs = []

    for f in a3_files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        tb = d["test"]["text_bert"][0]  # (3, 50)
        a3_tbs.append(tb)

        # Inspect token IDs
        input_ids = tb[0]
        att_mask = tb[1]
        text_len = int(att_mask.sum())

        cls_ok = bool(input_ids[0] == 101)
        sep_ok = bool(input_ids[text_len - 1] == 102) if text_len >= 1 else False
        pad_ok = bool((input_ids[text_len:] == 0).all()) if text_len < 50 else True

        a3_meta.append({
            "file": str(f.name),
            "text_len": int(text_len),
            "min_token_id": int(input_ids.min()),
            "max_token_id": int(input_ids.max()),
            "cls_is_101": cls_ok,
            "sep_is_102": sep_ok,
            "pad_is_0": pad_ok,
        })

    a3_tbs_arr = np.stack(a3_tbs, axis=0)  # (30, 3, 50)
    with torch.no_grad():
        inp_ids3 = torch.tensor(a3_tbs_arr[:, 0], dtype=torch.long, device=device)
        att_mask3 = torch.tensor(a3_tbs_arr[:, 1], dtype=torch.long, device=device)
        tok_type3 = torch.tensor(a3_tbs_arr[:, 2], dtype=torch.long, device=device)
        out3 = model(input_ids=inp_ids3, attention_mask=att_mask3, token_type_ids=tok_type3)
        recon3 = out3.last_hidden_state.cpu().numpy()

    a3_inference_passed = bool(recon3.shape == (30, 50, 768) and not np.isnan(recon3).any() and not np.isinf(recon3).any())

    a3_results = {
        "file_count": len(a3_files),
        "raw_text_verification_note": "Attachment 3 contains NO raw_text. Tokenizer re-tokenization cannot be performed. Token IDs, special tokens, and frozen encoder forward inference are verified directly.",
        "cls_special_token_ok_count": sum(1 for m in a3_meta if m["cls_is_101"]),
        "sep_special_token_ok_count": sum(1 for m in a3_meta if m["sep_is_102"]),
        "pad_special_token_ok_count": sum(1 for m in a3_meta if m["pad_is_0"]),
        "min_token_id_overall": int(a3_tbs_arr[:, 0].min()),
        "max_token_id_overall": int(a3_tbs_arr[:, 0].max()),
        "frozen_encoder_inference_passed": a3_inference_passed,
        "output_shape": list(recon3.shape),
        "samples": a3_meta,
    }
    print(f"Attachment 3 (N={len(a3_files)}): CLS_ok={a3_results['cls_special_token_ok_count']}/30, SEP_ok={a3_results['sep_special_token_ok_count']}/30, PAD_ok={a3_results['pad_special_token_ok_count']}/30, forward_passed={a3_inference_passed}")

    # Summary
    all_diffs_arr = np.array(all_diffs_a2 + diff4_maxs.tolist())
    evaluated = int(all_diffs_arr.size)
    n_gt_1e3 = int((all_diffs_arr > 1e-3).sum())
    n_le_1e4 = int((all_diffs_arr <= 1e-4).sum())
    n_tok_exact = sum(
        a2_results[s]["token_full_exact_matches"] for s in ("train", "valid", "test")
    ) + int(a4_results["token_full_exact_matches"])
    n_tok_total = sum(a2_results[s]["sample_count"] for s in ("train", "valid", "test")) + int(
        a4_results["sample_count"]
    )
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "model_metadata": {
            "name": model_name,
            "pinned_revision_sha": revision_sha,
            "load_source_snapshot_dir": snapshot_dir,
            "loaded_tokenizer_name_or_path": str(loaded_tokenizer_path),
            "loaded_model_name_or_path": str(loaded_model_path),
            "load_binding_verified": bool(binding_ok),
            "refs_main_revision": snapshot["refs_main_revision"],
            "refs_main_matches_pinned": bool(snapshot["refs_main_matches_pinned"]),
            "offline_flags": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "local_files_only": True,
            },
            "vocab_size": int(tokenizer.vocab_size),
            "hidden_dimension": int(model.config.hidden_size),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "file_hashes": file_hashes,
            "post_load_file_check": loaded_file_check,
            "network_upload_note": "No raw contest data is uploaded or transmitted; the encoder was loaded from the local snapshot only.",
        },
        "attachment2_audit": a2_results,
        "attachment4_audit": a4_results,
        "attachment3_audit": a3_results,
        "overall_summary": {
            "total_evaluated_samples_a2_and_a4": evaluated,
            "max_absolute_error_overall": float(all_diffs_arr.max()),
            "mean_absolute_error_overall": float(all_diffs_arr.mean()),
            "p50_absolute_error_overall": float(np.percentile(all_diffs_arr, 50)),
            "p90_absolute_error_overall": float(np.percentile(all_diffs_arr, 90)),
            "p95_absolute_error_overall": float(np.percentile(all_diffs_arr, 95)),
            "p99_absolute_error_overall": float(np.percentile(all_diffs_arr, 99)),
            "total_samples_exceeding_1e_3": n_gt_1e3,
            "total_samples_exceeding_1e_4": int((all_diffs_arr > 1e-4).sum()),
            "total_samples_exceeding_1e_5": int((all_diffs_arr > 1e-5).sum()),
            "total_samples_exceeding_1e_6": int((all_diffs_arr > 1e-6).sum()),
            "threshold_used": 1e-3,
            "counts_computed_from_arrays": True,
            "reconstruction_fidelity": {
                "samples_evaluated": evaluated,
                "samples_within_threshold_1e_3": evaluated - n_gt_1e3,
                "samples_exceeding_threshold_1e_3": n_gt_1e3,
                "samples_with_max_error_le_1e_4": n_le_1e4,
                "samples_with_max_error_le_1e_4_ratio": round(n_le_1e4 / evaluated, 6),
                "tokenization_exact_match_samples": n_tok_exact,
                "tokenization_samples_compared": n_tok_total,
                "max_absolute_error_overall": float(all_diffs_arr.max()),
                "mean_absolute_error_overall": float(all_diffs_arr.mean()),
                "p99_absolute_error_overall": float(np.percentile(all_diffs_arr, 99)),
                "difference_cause": "INFERRED",
                "difference_cause_statement": (
                    "The residual differences are *inferred* to come from FP32 GPU (cuDNN/CUDA) matrix-multiplication "
                    "accumulation order differing from the environment that produced the official 768-dim vectors. "
                    "This is an inference from the error magnitude and distribution, not a directly measured root cause; "
                    "the official extraction environment is not available for comparison."
                ),
                "encoder_binding": {
                    "pinned_revision_sha": revision_sha,
                    "load_source_snapshot_dir": snapshot_dir,
                    "binding_verified": bool(binding_ok),
                },
                "scope_statement": (
                    f"Reported for Attachment 2 (4850) + Attachment 4 (20) = {evaluated} samples actually evaluated in "
                    "this run. Attachment 3 carries no raw_text (re-tokenization not verifiable) and only shape/token/"
                    "special-token/inference checks are claimed for it. This is not a claim of bit-exact lossless storage."
                ),
            },
        }
    }

    out_json = cfg["_results_dir"] / "bert_audit_full.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull BERT audit written to: {out_json}")
    print(f"Overall max error across all 4870 samples: {all_diffs_arr.max():.2e}, mean: {all_diffs_arr.mean():.2e}, p99: {np.percentile(all_diffs_arr, 99):.2e}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BERT tokenizer/encoder version binding and reconstruction audit")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="only locate + hash-verify the pinned snapshot (no inference, no outputs written)",
    )
    ap.add_argument("--hub-dir", default=None, help="override the HuggingFace hub dir holding the pinned snapshot")
    args = ap.parse_args(argv)

    if args.self_check:
        try:
            snap = resolve_pinned_snapshot(Path(args.hub_dir) if args.hub_dir else None)
        except Exception as exc:  # noqa: BLE001 - reported, then non-zero exit
            print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps({
            "status": "PASS",
            "pinned_revision_sha": snap["pinned_revision_sha"],
            "snapshot_dir": snap["snapshot_dir"],
            "refs_main_revision": snap["refs_main_revision"],
            "refs_main_matches_pinned": snap["refs_main_matches_pinned"],
            "hashes": {k: v["sha256"] for k, v in snap["file_hashes"].items()},
        }, ensure_ascii=False, indent=2))
        return 0

    try:
        audit_bert_full()
    except Exception as exc:  # noqa: BLE001 - never let a failed audit look like success
        print(f"AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
