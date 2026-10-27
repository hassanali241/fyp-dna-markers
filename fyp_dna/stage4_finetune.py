"""
Stage 4: fine-tune the pretrained DNAFormer on ONE arm, then score the best checkpoint on
the test sets. Each team member runs this for a different arm with identical settings.

Training plan (identical for all arms; only --arm changes)
  start      pretrained DNAFormer (strict load, renamed fusion weights mapped)
  data       the arm's 10,000 training clusters (1/3/5/8% deletion mixed), relaxed read policy
  loss       DNAFormer's own: cross-entropy + 0.25 x consistency (left/right siamese branches)
  optimizer  Adam (betas 0.9/0.999, eps 1e-8, no weight decay), as in DNAFormer
  phase A    epochs 1-2: transformer frozen; alignment, embedding, output and fusion layers
             train at constant lr 1e-5 (adapts the input side to deletion-heavy reads)
  phase B    epochs 3-10: whole model trains; 100-step warm-up to 3e-6, then cosine decay to
             1e-7. 3e-6 is DNAFormer's peak lr 1e-5 scaled to our batch size (16 vs 64).
  batch      16 clusters; mixed precision (fp16) with gradient scaling; grad-norm clip 1.0
  budget     fixed 10 epochs x 625 steps for every arm (no early stop, so budgets are equal)
  selection  best epoch by payload-only validation normalized edit distance (val split of
             the same arm). The test sets are touched only once, after training.
  resume     last.pt is saved after every epoch; rerunning continues from it. Settings are
             stored in the checkpoint and a resume with different settings is refused.

Outputs
  <ckpt-dir>/<arm>/last.pt, best.pt, history.json, train_log.txt
  <out-dir>/<arm>/predictions.csv, results.csv, summary.json   (same format as Stage 3,
                                                                 method = dnaformer_ft)
Usage
  python stage4_finetune.py --repo R --data D --ckpt PRETRAINED --arm marker_periodic \
      --ckpt-dir CKDIR --out-dir OUTDIR
  add --smoke for a 2-minute test (tiny epochs, nothing written to Drive folders you pass)
"""
import argparse
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dnaf_adapter as A  # noqa: E402
import stage1_generate as G  # noqa: E402
import stage3_baselines as S  # noqa: E402

SETTINGS = {
    "seed": 1,
    "batch_size": 16,
    "epochs": 10,
    "frozen_epochs": 2,
    "lr_frozen": 1e-5,
    "lr_peak": 3e-6,
    "lr_min": 1e-7,
    "warmup_steps": 100,
    "ce_coeff": 1.0,
    "consistency_coeff": 0.25,
    "grad_clip": 1.0,
    "amp_fp16": True,
    "read_policy": "relaxed",
    "steps_per_epoch": None,     # None = full pass over the training set (625 steps)
    "val_limit": None,           # None = all 1,000 validation clusters
    "selection_metric": "val_ned",
}
SMOKE = {"epochs": 3, "frozen_epochs": 1, "steps_per_epoch": 20, "warmup_steps": 5, "val_limit": 64}

LOG_FILE = None


def log(msg):
    print(msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(data, arm, subset, limit=None):
    path = os.path.join(data, arm, subset, "metadata.json")
    if not os.path.isfile(path):
        sys.exit("ERROR: dataset file not found: %s\nThe Stage 1 data is missing from Google Drive "
                 "(MyDrive/fyp_dna/data/stage1_pilot). Put it back, remount Drive, and run again." % path)
    with open(path) as f:
        clusters = json.load(f)
    return clusters[:limit] if limit else clusters


def epoch_order(n, seed, epoch):
    """Same shuffled order for every arm (the arms are paired), reproducible after resume."""
    return np.random.default_rng([seed, epoch]).permutation(n)


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------
def set_phase(model, phase):
    """Phase A: freeze the transformer encoder. Phase B: everything trainable."""
    for name, p in model.named_parameters():
        p.requires_grad = (phase == "B") or not name.startswith("encoder.")
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(torch, params, lr):
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def lr_at(cfg, epoch, step_in_epoch, steps_per_epoch):
    """Learning rate for a given (1-based) epoch and 0-based step."""
    if epoch <= cfg["frozen_epochs"]:
        return cfg["lr_frozen"]
    b_steps = (cfg["epochs"] - cfg["frozen_epochs"]) * steps_per_epoch
    t = (epoch - cfg["frozen_epochs"] - 1) * steps_per_epoch + step_in_epoch
    if t < cfg["warmup_steps"]:
        return cfg["lr_peak"] * (t + 1) / cfg["warmup_steps"]
    frac = (t - cfg["warmup_steps"]) / max(1, b_steps - cfg["warmup_steps"])
    return cfg["lr_min"] + 0.5 * (cfg["lr_peak"] - cfg["lr_min"]) * (1 + math.cos(math.pi * min(1.0, frac)))


def dnaformer_loss(torch, out, y, cfg):
    """DNAFormer's loss (DNN/loss.py): CE on the fused output + consistency CE on each branch.

    As in the original, the consistency term applies cross-entropy to the softmax of each
    branch with the one-hot label as a probability target."""
    F = torch.nn.functional
    onehot = F.one_hot(y, 4).permute(0, 2, 1).float()        # [B, 4, 128]
    ce = F.cross_entropy(out["pred"].float(), y)
    cons = 0.5 * (F.cross_entropy(out["pred_left"].float().softmax(dim=1), onehot) +
                  F.cross_entropy(out["pred_right"].float().softmax(dim=1), onehot))
    return cfg["ce_coeff"] * ce + cfg["consistency_coeff"] * cons, ce, cons


def evaluate_val(torch, model, arm, clusters, device):
    """Payload-only NED and exact rate on validation, overall and per deletion level."""
    model.eval()
    preds, _, _ = S.run_dnaformer(model, arm, clusters, device)
    per_level = {}
    for c, p in zip(clusters, preds):
        s = S.score(p, c["payload_strand"])
        per_level.setdefault(c["deletion_rate"], []).append((s["ned"], s["exact"]))
    allv = [v for vs in per_level.values() for v in vs]
    res = {"ned": float(np.mean([v[0] for v in allv])), "exact": float(np.mean([v[1] for v in allv]))}
    for d, vs in sorted(per_level.items()):
        res["ned_%s" % G.level_tag(d)] = float(np.mean([v[0] for v in vs]))
        res["exact_%s" % G.level_tag(d)] = float(np.mean([v[1] for v in vs]))
    return res


def fmt_val(v):
    lv = " ".join("%s %.1f%%/%.4f" % (k[6:], 100 * v[k], v["ned_" + k[6:]])
                  for k in sorted(v) if k.startswith("exact_del"))
    return "NED %.4f exact %.1f%% | per level exact/NED: %s" % (v["ned"], 100 * v["exact"], lv)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_atomic(torch, obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state(torch):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(torch, st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    # RNG states must be CPU byte tensors; torch.load(map_location='cuda') moves them to the GPU
    torch.set_rng_state(st["torch"].cpu())
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global LOG_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True, help="pretrained DNAFormer_siamese.pth")
    ap.add_argument("--arm", required=True, choices=G.ALL_ARMS)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="2-minute test run")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after this epoch (to test resume)")
    ap.add_argument("--allow-cpu", action="store_true", help="for code testing only")
    args = ap.parse_args()

    import torch
    cfg = dict(SETTINGS)
    if args.smoke:
        cfg.update(SMOKE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        sys.exit("ERROR: no GPU. Runtime > Change runtime type > T4 GPU, then run again.")

    ck_dir = os.path.join(args.ckpt_dir, args.arm)
    out_dir = os.path.join(args.out_dir, args.arm)
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    LOG_FILE = os.path.join(ck_dir, "train_log.txt")
    last_path, best_path = os.path.join(ck_dir, "last.pt"), os.path.join(ck_dir, "best.pt")
    hist_path = os.path.join(ck_dir, "history.json")

    log("=" * 70)
    log("Stage 4 fine-tuning | arm %s | %s" % (args.arm, time.strftime("%Y-%m-%d %H:%M:%S")))
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    log("python %s, torch %s, device %s" % (platform.python_version(), torch.__version__, gpu))
    log("settings: %s" % json.dumps(cfg))

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train = load_split(args.data, args.arm, "train")
    val = load_split(args.data, args.arm, "val", cfg["val_limit"])
    spe = cfg["steps_per_epoch"] or len(train) // cfg["batch_size"]
    log("train clusters %d, val clusters %d, %d steps/epoch x %d epochs"
        % (len(train), len(val), spe, cfg["epochs"]))

    if not os.path.isfile(args.ckpt):
        sys.exit("ERROR: pretrained model not found: %s\nPut DNAFormer_siamese.pth back in "
                 "MyDrive/fyp_dna/checkpoints (or rerun the Stage 2 notebook), then run again." % args.ckpt)
    model, _ = A.build_model(args.repo, device)
    rep = A.load_checkpoint(model, args.ckpt, device)
    log("pretrained checkpoint loaded strictly (%d tensors)" % rep["n_tensors"])
    use_amp = cfg["amp_fp16"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, history, best = 1, [], {"epoch": 0, "val_ned": float("inf")}
    if os.path.isfile(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        if st["settings"] != cfg or st["arm"] != args.arm:
            sys.exit("ERROR: %s was made with different settings or arm; refusing to resume.\n"
                     "Saved: %s %s" % (last_path, st["arm"], st["settings"]))
        model.load_state_dict(st["model"], strict=True)
        start_epoch, history, best = st["epoch"] + 1, st["history"], st["best"]
        log("RESUMING from %s: finished epoch %d, best epoch %d" % (last_path, st["epoch"], best["epoch"]))
    else:
        st = None

    if start_epoch == 1 and not history:
        t0 = time.time()
        v0 = evaluate_val(torch, model, args.arm, val, device)
        history.append({"epoch": 0, "phase": "pretrained", "val": v0, "train_loss": None,
                        "time_s": time.time() - t0})
        best = {"epoch": 0, "val_ned": v0["ned"]}
        log("epoch 0 (pretrained, no training): %s" % fmt_val(v0))

    phase = None
    opt = None
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        new_phase = "A" if epoch <= cfg["frozen_epochs"] else "B"
        if new_phase != phase:
            params = set_phase(model, new_phase)
            opt = make_optimizer(torch, params, lr_at(cfg, epoch, 0, spe))
            n_tr = sum(p.numel() for p in params)
            log("phase %s: %s, %.1fM trainable parameters" % (
                new_phase, "transformer frozen" if new_phase == "A" else "whole model", n_tr / 1e6))
            phase = new_phase
            if st is not None and st.get("phase") == phase:
                opt.load_state_dict(st["optimizer"])
                scaler.load_state_dict(st["scaler"])
                log("optimizer state restored")
        if st is not None:
            set_rng_state(torch, st["rng"])
        st = None

        model.train()
        order = epoch_order(len(train), cfg["seed"], epoch)
        t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sums = np.zeros(3)
        n_skipped = 0
        for step in range(spe):
            for g in opt.param_groups:
                g["lr"] = lr_at(cfg, epoch, step, spe)
            idx = order[(step * cfg["batch_size"]) % len(train):][:cfg["batch_size"]]
            batch = [train[i] for i in idx]
            x, y = A.make_batch(batch, cfg["read_policy"], device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss, ce, cons = dnaformer_loss(torch, out, y, cfg)
            if not torch.isfinite(loss):
                n_skipped += 1
                if n_skipped > 20:
                    sys.exit("ERROR: loss is NaN/inf for more than 20 steps; stopping. Paste this to Claude.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sums += [loss.item(), ce.item(), cons.item()]
            if (step + 1) % max(1, spe // 5) == 0 or step + 1 == spe:
                el = time.time() - t0
                left = el / (step + 1) * (spe - step - 1)
                log("  epoch %d step %d/%d  loss %.4f (ce %.4f, cons %.4f)  lr %.2e  %.2fs/step  epoch ends in %.0f min"
                    % (epoch, step + 1, spe, sums[0] / (step + 1), sums[1] / (step + 1), sums[2] / (step + 1),
                       opt.param_groups[0]["lr"], el / (step + 1), left / 60))
        train_time = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

        v = evaluate_val(torch, model, args.arm, val, device)
        rec = {"epoch": epoch, "phase": phase, "val": v, "train_loss": sums[0] / max(1, spe - n_skipped),
               "skipped_steps": n_skipped, "time_s": time.time() - t0, "peak_gpu_gb": peak}
        history.append(rec)
        improved = v["ned"] < best["val_ned"]
        if improved:
            best = {"epoch": epoch, "val_ned": v["ned"]}
            save_atomic(torch, {"model": model.state_dict(), "epoch": epoch, "arm": args.arm,
                                "settings": cfg, "val": v}, best_path)
        log("epoch %d done in %.1f min (peak GPU %.1f GB): %s%s" % (
            epoch, rec["time_s"] / 60, peak, fmt_val(v), "  <- best so far, saved" if improved else ""))
        save_atomic(torch, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "scaler": scaler.state_dict(), "rng": rng_state(torch), "phase": phase,
                            "epoch": epoch, "history": history, "best": best, "arm": args.arm,
                            "settings": cfg}, last_path)
        with open(hist_path, "w") as f:
            json.dump({"arm": args.arm, "settings": cfg, "history": history, "best": best}, f, indent=1)
        if args.stop_after and epoch >= args.stop_after:
            log("stopping after epoch %d as asked (--stop-after); run again to resume" % epoch)
            return

    # ------------------------------------------------------------------ test evaluation
    log("\nTraining finished. Best epoch %d (val NED %.4f)." % (best["epoch"], best["val_ned"]))
    if best["epoch"] > 0:
        bst = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(bst["model"], strict=True)
    else:
        log("NOTE: no epoch beat the pretrained model on validation; scoring the pretrained weights.")
        A.load_checkpoint(model, args.ckpt, device)
    model.eval()
    rows, results = [], []
    for level in G.DELETION_LEVELS:
        clusters = S.load_test(args.data, args.arm, level, 25 if args.smoke else 0)
        preds, runtime, peak = S.run_dnaformer(model, args.arm, clusters, device)
        r = S.make_rows("dnaformer_ft", args.arm, level, clusters, preds)
        rows += r
        results.append(S.summarize("dnaformer_ft", args.arm, level, r, runtime, peak))
    S.write_csv(os.path.join(out_dir, "predictions.csv"), rows, S.PRED_FIELDS)
    S.write_csv(os.path.join(out_dir, "results.csv"), results, list(results[0].keys()))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"arm": args.arm, "best_epoch": best["epoch"], "best_val_ned": best["val_ned"],
                   "settings": cfg, "gpu": gpu, "torch": torch.__version__, "smoke": args.smoke,
                   "history": history}, f, indent=1)
    log("\nTEST RESULTS (fine-tuned DNAFormer, %s, best epoch %d):" % (args.arm, best["epoch"]))
    S.print_table(results)
    log("\nSaved to %s" % out_dir)
    log("STAGE 4 DONE (%s)" % args.arm)


if __name__ == "__main__":
    main()
