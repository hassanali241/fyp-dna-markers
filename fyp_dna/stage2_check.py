"""
Stage 2: DNAFormer compatibility check. No training.

Checks that:
  A. the environment and GPU are usable
  B. the pretrained checkpoint loads into the unchanged model with NO missing weights,
     and its layer sizes confirm input 132 / output 128
  C. the repo's own data loader has the problems we expect (so we don't use it)
  D. our adapter turns one marker-free, one periodic and one end-block cluster into
     correctly shaped input
  E. the pretrained model runs on them and gives 128-base output
  F. a small sanity inference: pretrained model on a few test clusters per level
  G. GPU memory and speed for one fine-tuning step at several batch sizes

Usage:
  python stage2_check.py --repo REPO --data DATA --ckpt CKPT [--n-sanity 50] [--log out.json]
  python stage2_check.py --repo REPO --data DATA --no-torch     # only the parts without PyTorch
"""
import argparse
import json
import os
import platform
import random
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dnaf_adapter as A  # noqa: E402
import stage1_generate as G  # noqa: E402

FAILS = []
SUMMARY = {}


def check(ok, msg):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAILS.append(msg)


def load_subset(data, arm, subset):
    with open(os.path.join(data, arm, subset, "metadata.json")) as f:
        return json.load(f)


def score(arm, pred, cluster):
    payload = cluster["payload_strand"]
    pred_payload = G.remove_markers(arm, pred)
    ed = A.edit_distance(pred_payload, payload)
    return ed / len(payload), pred_payload == payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-sanity", type=int, default=50, help="test clusters per level for the sanity inference")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    levels = G.DELETION_LEVELS
    tests = ["test_" + G.level_tag(d) for d in levels]

    # ------------------------------------------------------------------ A
    print("A. Environment")
    print("   python %s" % platform.python_version())
    SUMMARY["python"] = platform.python_version()
    device = "cpu"
    torch = None
    if not args.no_torch:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("   torch %s, device %s" % (torch.__version__, device))
        SUMMARY["torch"] = torch.__version__
        if device == "cuda":
            p = torch.cuda.get_device_properties(0)
            print("   GPU %s, %.1f GB" % (p.name, p.total_memory / 1e9))
            SUMMARY["gpu"] = p.name
            SUMMARY["gpu_memory_gb"] = round(p.total_memory / 1e9, 1)
        else:
            print("   WARNING: no GPU. Switch Runtime > Change runtime type > T4 GPU.")
        torch.manual_seed(0)
    random.seed(0)

    # ------------------------------------------------------------------ C (no torch needed)
    print("\nC. Problems in the repo's own data loader")
    try:
        random.shuffle([1, 2, 3], random.random)
        shuffle_ok = True
    except TypeError:
        shuffle_ok = False
    print("   repo read_data() calls random.shuffle(x, random.random): %s on Python %s"
          % ("works" if shuffle_ok else "CRASHES (TypeError)", platform.python_version()))
    SUMMARY["repo_shuffle_crashes"] = not shuffle_ok

    if not args.no_torch:
        try:
            sys.path.insert(0, args.repo)
            from DNN import data_loader as repo_dl  # noqa: E402
            fake = types.SimpleNamespace(config=A.DNAFormerConfig())
            reads = ["A" * 132, "C" * 128]
            left, _, _, _ = repo_dl.grab_model_input(fake, {"noisy_copies": reads})
            row132, row128 = float(left[0].sum()), float(left[1].sum())
            print("   repo grab_model_input(): read of length 132 -> %d non-zero bases; length 128 -> %d"
                  % (row132, row128))
            SUMMARY["repo_len132_becomes_zero"] = row132 == 0
        except Exception as e:  # noqa: BLE001
            print("   (could not import repo data_loader to demonstrate: %s)" % e)

    print("   repo length filter (|len-128|<=4) on our test sets; a cluster with 0 reads left is")
    print("   silently REPLACED by a random other cluster in the repo loader:")
    print("   arm              level  reads kept (of 10)  clusters swapped")
    SUMMARY["repo_filter"] = {}
    for arm in ["marker_free", "marker_periodic"]:
        for t, d in zip(tests, levels):
            cl = load_subset(args.data, arm, t)
            kept = [len(A.select_reads(c["noisy_reads"], "original")) for c in cl]
            swapped = float(np.mean([k == 0 for k in kept]))
            print("   %-16s %4.0f%%   %5.2f               %5.1f%%" % (arm, d * 100, np.mean(kept), 100 * swapped))
            SUMMARY["repo_filter"]["%s_%s" % (arm, t)] = {"mean_kept": round(float(np.mean(kept)), 2),
                                                          "swapped_frac": round(swapped, 3)}

    # ------------------------------------------------------------------ D (no torch needed)
    print("\nD. Adapter encoding of one cluster per arm (test_del05, pair 0)")
    examples = {arm: load_subset(args.data, arm, "test_del05")[0] for arm in G.ARMS}
    check(len({c["pair_id"] for c in examples.values()}) == 1, "the three example clusters are the same pair")
    for arm, c in examples.items():
        left, right, n = A.encode_cluster(c["noisy_reads"], "relaxed")
        lens = [len(r) for r in c["noisy_reads"]]
        filled = left.sum(axis=(1,))  # [16, 132], 1 where a base is present
        ok = (left.shape == (16, 4, 132) and right.shape == (16, 4, 132) and n == 10 and
              all(int(filled[i].sum()) == min(lens[i], 132) for i in range(10)) and
              filled[10:].sum() == 0 and
              all(A.one_hot(c["noisy_reads"][i][::-1]).tolist() == right[i].tolist() for i in range(10)))
        check(ok, "%-16s input [16,4,132] x2, 10 reads used, slots 11-16 empty, read lengths %d-%d"
              % (arm, min(lens), max(lens)))
        check(len(c["clean_strand"]) == A.LABEL_LEN, "%-16s target length 128 matches model output" % arm)

    if args.no_torch:
        finish(args)
        return

    # ------------------------------------------------------------------ B
    print("\nB. Pretrained checkpoint")
    model, cfg = A.build_model(args.repo, device)
    n_params = sum(p.numel() for p in model.parameters())
    print("   model built from repo code: %.1fM parameters" % (n_params / 1e6))
    SUMMARY["n_params_M"] = round(n_params / 1e6, 1)
    if not args.ckpt or not os.path.isfile(args.ckpt):
        check(False, "checkpoint file found at %s" % args.ckpt)
        finish(args)
        return
    try:
        rep = A.load_checkpoint(model, args.ckpt, device)
        check(True, "all %d weight tensors loaded strictly (0 missing, 0 unexpected, 0 wrong shape) [%s]"
              % (rep["n_tensors"], rep["loader"]))
        print("   checkpoint keys: %s, epoch: %s" % (rep["checkpoint_top_keys"], rep["checkpoint_epoch"]))
        for old, new in rep["renamed"].items():
            print("   renamed %s -> %s, shape %s, min/mean/max %s"
                  % (old, new, rep["renamed_shapes"][new], rep["renamed_stats"][new]))
        SUMMARY["checkpoint"] = {k: rep[k] for k in ("loader", "n_tensors", "checkpoint_epoch", "renamed",
                                                     "renamed_stats")}
    except RuntimeError as e:
        check(False, str(e)[:500])
        finish(args)
        return
    sd = model.state_dict()
    shapes = {
        "alignement.linear_block.fc_1.weight": (132, 132),
        "embedding.linear_block.fc_1.weight": (128, 132),
        "fusion.pred_fusion_left": (128,),
    }
    for k, want in shapes.items():
        check(tuple(sd[k].shape) == want, "%s has shape %s (input 132 -> output 128)" % (k, tuple(sd[k].shape)))

    # ------------------------------------------------------------------ E
    print("\nE. Forward pass on the three example clusters (pretrained, no fine-tuning)")
    model.eval()
    for arm, c in examples.items():
        x, y = A.make_batch([c], "relaxed", device)
        with torch.inference_mode():
            out = model(x)
        shp = tuple(out["pred"].shape)
        check(shp == (1, 4, 128), "%-16s output shape %s" % (arm, shp))
        pred = A.decode(out["pred"])[0]
        ned, exact = score(arm, pred, c)
        print("     payload-only normalized edit distance %.3f, exact=%s" % (ned, exact))

    # ------------------------------------------------------------------ F
    print("\nF. Sanity inference: pretrained DNAFormer, %d test clusters per level" % args.n_sanity)
    print("   (only a sanity check; the proper baseline evaluation is Stage 3)")
    print("   arm / read policy            level  exact   norm. edit dist")
    runs = [("marker_free", "original"), ("marker_free", "relaxed"),
            ("marker_periodic", "relaxed"), ("marker_endblock", "relaxed"),
            ("marker_free", "relaxed+fusion=1")]   # last: repo behaviour (renamed weights skipped)
    SUMMARY["sanity"] = {}
    saved_fusion = None
    for arm, pol in runs:
        if pol.endswith("fusion=1") and saved_fusion is None:
            saved_fusion = (model.fusion.pred_fusion_left.detach().clone(),
                            model.fusion.pred_fusion_right.detach().clone())
            A.reset_fusion_weights(model)
        pol_read = pol.split("+")[0]
        for t, d in zip(tests, levels):
            cl = load_subset(args.data, arm, t)[:args.n_sanity]
            neds, exacts = [], []
            for i in range(0, len(cl), 25):
                chunk = cl[i:i + 25]
                x, _ = A.make_batch(chunk, pol_read, device)
                with torch.inference_mode():
                    preds = A.decode(model(x)["pred"])
                for p, c in zip(preds, chunk):
                    ned, ex = score(arm, p, c)
                    neds.append(ned)
                    exacts.append(ex)
            print("   %-16s %-16s %4.0f%%  %5.1f%%  %.4f" % (arm, pol, d * 100, 100 * np.mean(exacts), np.mean(neds)))
            SUMMARY["sanity"]["%s_%s_%s" % (arm, pol, t)] = {"exact": round(float(np.mean(exacts)), 3),
                                                            "ned": round(float(np.mean(neds)), 4)}
    if saved_fusion is not None:
        with torch.no_grad():
            model.fusion.pred_fusion_left.copy_(saved_fusion[0])
            model.fusion.pred_fusion_right.copy_(saved_fusion[1])
    mapped = np.mean([SUMMARY["sanity"]["marker_free_relaxed_%s" % t]["ned"] for t in tests])
    ones = np.mean([SUMMARY["sanity"]["marker_free_relaxed+fusion=1_%s" % t]["ned"] for t in tests])
    print("   mean norm. edit dist over levels: renamed weights loaded %.4f vs fusion=1 (repo) %.4f" % (mapped, ones))
    SUMMARY["fusion_choice"] = "renamed" if mapped <= ones else "ones"
    mf1 = SUMMARY["sanity"]["marker_free_relaxed_test_del01"]["ned"]
    check(mf1 < 0.05, "pretrained model is clearly working at 1%% deletion (norm. edit dist %.4f < 0.05)" % mf1)

    # ------------------------------------------------------------------ G
    print("\nG. One fine-tuning step: GPU memory and speed (full model, Adam, fp32)")
    if device != "cuda":
        print("   skipped (no GPU)")
    else:
        train = load_subset(args.data, "marker_free", "train")
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        ce = torch.nn.CrossEntropyLoss()
        SUMMARY["memory"] = {}
        for bs in [8, 16, 32, 64]:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x, y = A.make_batch(train[:bs], "relaxed", device)
                times = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    opt.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = ce(out["pred"], y)
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
                peak = torch.cuda.max_memory_allocated() / 1e9
                print("   batch %3d: peak %.2f GB, %.3f s/step" % (bs, peak, float(np.median(times))))
                SUMMARY["memory"][bs] = {"peak_gb": round(peak, 2), "s_per_step": round(float(np.median(times)), 3)}
            except torch.cuda.OutOfMemoryError:
                print("   batch %3d: OUT OF MEMORY" % bs)
                SUMMARY["memory"][bs] = "OOM"
                break
            finally:
                opt.zero_grad(set_to_none=True)
        print("   (weights were changed by these test steps; nothing is saved)")

    finish(args)


def finish(args):
    SUMMARY["fails"] = FAILS
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, "w") as f:
            json.dump(SUMMARY, f, indent=1)
    print("\n" + ("ALL STAGE 2 CHECKS PASSED" if not FAILS else
                  "%d CHECK(S) FAILED:\n  - " % len(FAILS) + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
