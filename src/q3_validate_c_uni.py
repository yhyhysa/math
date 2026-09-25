"""Recheck Q3's reused model and original-word evidence on A2 valid only.

Run from 工程: python -m src.q3_validate_c_uni --tag my_run
This script never trains or changes a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel

from src.config import load_config
from src.data import load_pickle
from src.q2_train_baseline import BERT_REV, metrics
from src.q2_train_transformer import recode_text
from src.q3_explain_c_uni import (load_models, make_view, model_scores, one_view,
                                  replace, text_reference, windows,
                                  words_and_tokens, write_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag can contain only letters, numbers, _ and -")
    cfg = load_config()
    root = cfg["_engine_dir"]
    out = root / "results" / f"q3_c_uni_validation_{args.tag}"
    if out.exists():
        raise FileExistsError(f"Preserving existing run: {out}")
    out.mkdir(parents=True)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models, checkpoint = load_models(root, device)
        path = next((root / "data/raw/extracted/E题数据/附件2-数据集特征文件").rglob("aligned_50.pkl"))
        samples = load_pickle(path).splits["valid"]
        if len(samples) != 728:
            raise ValueError("Expected 728 A2 validation samples")
        view = make_view(samples, checkpoint)
        probs, values = model_scores(models, [one_view(view, i) for i in range(len(samples))], device)
        labels = np.array([int(s.classification_label) for s in samples], dtype=np.int64)
        true_values = np.array([float(s.regression_label) for s in samples], dtype=np.float32)
        full_metrics = metrics(labels, true_values, probs, values)
        print("A2 valid 728:", full_metrics, flush=True)
        rng = np.random.default_rng(20260925)
        picks = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == cls), size=count, replace=False)
                                        for cls, count in ((0, 43), (1, 42), (2, 43))]))
        bert_dir = root / "models/bert-base-uncased" / BERT_REV
        tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True, use_fast=True)
        bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
        bert.requires_grad_(False)
        text_ref = text_reference(root)
        audit, recode_masks, recode_views = [], [], []
        for count, i in enumerate(picks, 1):
            sample = samples[int(i)]
            original = one_view(view, int(i))
            pred_cls = int(probs[i].argmax())
            words = words_and_tokens(sample.raw_text, sample.arrays["text_bert"], tokenizer)
            possible = [w for w in windows(words) if w["word_end"]-w["word_start"] == min(3,len(words))]
            variants = []
            for w in possible:
                mask = np.zeros(50, dtype=bool)
                mask[w["indices"]] = True
                variants.append(replace(original, 0, mask, text_ref))
            perturbed, _ = model_scores(models, variants, device)
            drops = probs[i,pred_cls] - perturbed[:,pred_cls]
            best = int(np.argmax(drops))
            alternatives = [j for j,w in enumerate(possible) if j != best and
                            (w["word_end"] <= possible[best]["word_start"] or
                             w["word_start"] >= possible[best]["word_end"])]
            if not alternatives:
                alternatives = [j for j in range(len(possible)) if j != best]
            if not alternatives:
                continue
            random_j = int(rng.choice(alternatives))
            for kind,j in (("top",best),("random",random_j)):
                mask = np.zeros(50, dtype=bool)
                mask[possible[j]["indices"]] = True
                recode_masks.append(mask)
                recode_views.append(original)
                audit.append({"sample_id": sample.sample_id, "valid_index": int(i),
                              "kind": kind, "predicted_class": pred_cls,
                              "word_span": sample.raw_text[possible[j]["char_start"]:possible[j]["char_end"]],
                              "feature_level_class_drop": float(drops[j]),
                              "original_word_recode_class_drop": None})
            if count % 32 == 0:
                print(f"Ranked validation text windows {count}/128", flush=True)
        tb = np.stack([samples[row["valid_index"]].arrays["text_bert"] for row in audit]).astype(np.int64)
        masked_text = recode_text(tb, np.stack(recode_masks), bert, device)
        checked = []
        for base_view, text in zip(recode_views, masked_text):
            altered = {key: value.copy() for key,value in base_view.items()}
            altered["text"] = text
            checked.append(altered)
        new_probs, _ = model_scores(models, checked, device)
        for row, prob in zip(audit, new_probs):
            row["original_word_recode_class_drop"] = float(probs[row["valid_index"],row["predicted_class"]]-prob[row["predicted_class"]])
        pairs = list(zip(audit[::2], audit[1::2]))
        if not pairs:
            raise ValueError("No validation samples have two comparable word windows")
        top = np.array([a["original_word_recode_class_drop"] for a,_ in pairs])
        rand = np.array([b["original_word_recode_class_drop"] for _,b in pairs])
        summary = {"status": "COMPLETE", "model": "C_uni three-seed ensemble; no new training",
                   "valid_n": len(samples), "valid_metrics": full_metrics,
                   "word_review_n": len(pairs), "word_review_initial_n": len(picks),
                   "word_review_excluded_short_text": len(picks)-len(pairs),
                   "word_review_sampling": "fixed seed 20260925; initially 43 negative, 42 neutral, 43 positive by true valid label; clips without two word windows excluded",
                   "top_word_recode_drop_mean": float(top.mean()),
                   "random_word_recode_drop_mean": float(rand.mean()),
                   "top_beats_random_fraction": float(np.mean(top > rand)),
                   "top_positive_drop_fraction": float(np.mean(top > 0)),
                   "interpretation_limit": "Feature-level window ranking and original-word BERT re-encoding are different interventions. This validates model sensitivity, not human-ground-truth evidence localization or AV timestamps."}
        write_csv(out / "valid_word_check.csv", audit)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(out), **summary}, ensure_ascii=False), flush=True)
    except Exception as exc:
        (out / "STATUS.json").write_text(json.dumps({"status": "BLOCKED", "error": repr(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
