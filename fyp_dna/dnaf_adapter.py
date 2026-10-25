"""
Minimal adapter between our Stage 1 data and the UNCHANGED DNAFormer model.

What it does:
  - builds DNAFormer exactly as the repo defines it (DNN/model_DNAFormer_siamese.py)
  - loads the pretrained checkpoint and REFUSES to continue if any weight is missing
    (the repo uses strict=False, which silently leaves layers random)
  - turns a cluster of reads into the model's input tensors, without the repo
    loader's problems (crash on Python >= 3.11, reads of length 132 turned into
    zeros, deletion-heavy reads thrown away, clusters silently swapped)

The model architecture is not touched. Input/target lengths stay 132 / 128.
"""
import os
import sys

import numpy as np

BASES = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}   # same mapping as the repo: A=0 C=1 G=2 T=3

LABEL_LEN = 128        # repo: label_length 140 - index_length 12
INPUT_LEN = 132        # repo: noisy_copies_length = 140 + 4 - 12
MAX_READS = 16         # repo: max_number_per_cluster
ORIG_MAX_DEVIATION = 4  # repo: corrupt_max_deviation


class DNAFormerConfig:
    """Same values as DNN/inference.py in the repo."""
    index_length = 12
    label_length = 140
    filter_index = True
    corrupt_max_deviation = 4
    noisy_copies_length = INPUT_LEN
    max_number_per_cluster = MAX_READS
    read_padding = "end"
    model_config = "siamese"
    n_head = 32
    activation = "gelu"
    num_layers = 12
    d_model = 1024
    alignment_filters = 128
    alignement_filters = 128   # the repo's train.py uses this misspelling
    dim_feedforward = 2048
    output_ch = 4
    enc_filters = 4
    p_dropout = 0
    class_token = 0
    use_input_scaling = False
    device = "cpu"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(repo, device="cpu"):
    sys.path.insert(0, os.path.join(repo, "DNN"))
    from model_DNAFormer_siamese import net  # noqa: E402
    cfg = DNAFormerConfig()
    cfg.device = device
    return net(cfg).to(device), cfg


def load_checkpoint(model, path, device="cpu"):
    """Load weights strictly. Returns a report dict; raises if anything does not match."""
    import torch
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        loader = "weights_only=True"
    except Exception:
        # Official checkpoint from the DNAFormer authors; may contain non-tensor objects.
        ckpt = torch.load(path, map_location=device, weights_only=False)
        loader = "weights_only=False"
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_bad = sorted(k for k in model_keys & ckpt_keys
                       if tuple(model.state_dict()[k].shape) != tuple(state[k].shape))
    report = {
        "loader": loader,
        "checkpoint_top_keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else [],
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "n_tensors": len(ckpt_keys), "missing": missing, "unexpected": unexpected,
        "shape_mismatch": shape_bad,
    }
    if missing or unexpected or shape_bad:
        raise RuntimeError("Checkpoint does not match the model: %s" % report)
    model.load_state_dict(state, strict=True)
    return report


# ---------------------------------------------------------------------------
# Input encoding
# ---------------------------------------------------------------------------
def select_reads(reads, policy="relaxed"):
    """Which reads go into the model.

    relaxed  (ours): keep every read; reads longer than 132 are cut to 132.
    original (repo): keep reads with |len-128| <= 4; a read of exactly 132 becomes
                     an all-zero row (repo bug). If no read survives, the repo would
                     replace the whole cluster with a random other one.
    """
    if policy == "relaxed":
        return [r[:INPUT_LEN] for r in reads[:MAX_READS]]
    if policy == "original":
        kept = [r for r in reads if abs(len(r) - LABEL_LEN) <= ORIG_MAX_DEVIATION]
        return [r if len(r) < INPUT_LEN else "" for r in kept[:MAX_READS]]
    raise ValueError(policy)


def one_hot(read, length=INPUT_LEN):
    x = np.zeros((4, length), dtype=np.float32)
    for j, b in enumerate(read[:length]):
        x[BASE_IDX[b], j] = 1.0
    return x


def encode_cluster(reads, policy="relaxed"):
    """Return (left, right, n_used) as numpy arrays of shape [16, 4, 132].

    left  = each read one-hot, zero-padded at the end (repo read_padding='end')
    right = each read reversed, then zero-padded at the end (repo siamese branch)
    Unused read slots stay all-zero, as in the repo.
    """
    used = select_reads(reads, policy)
    left = np.zeros((MAX_READS, 4, INPUT_LEN), dtype=np.float32)
    right = np.zeros((MAX_READS, 4, INPUT_LEN), dtype=np.float32)
    for i, r in enumerate(used):
        left[i] = one_hot(r)
        right[i] = one_hot(r[::-1])
    return left, right, len(used)


def make_batch(clusters, policy="relaxed", device="cpu"):
    """clusters: list of Stage 1 metadata dicts. Returns (model_input, labels).

    model_input has shape [2B, 16, 4, 132]: left branch first, right branch second,
    exactly as the repo concatenates them. labels has shape [B, 128] (class indices).
    """
    import torch
    L, R, Y = [], [], []
    for c in clusters:
        left, right, _ = encode_cluster(c["noisy_reads"], policy)
        L.append(left)
        R.append(right)
        Y.append([BASE_IDX[b] for b in c["clean_strand"]])
    x = torch.from_numpy(np.concatenate([np.stack(L), np.stack(R)], axis=0)).to(device)
    y = torch.tensor(Y, dtype=torch.long, device=device)
    return x, y


def decode(pred_logits):
    """[B, 4, 128] logits -> list of DNA strings."""
    idx = pred_logits.argmax(dim=1).cpu().numpy()
    return ["".join(BASES[i] for i in row) for row in idx]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
