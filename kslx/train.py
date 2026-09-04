"""학습 + 프로토콜 비교.

python -m kslx.train --data data/kslx/word_271.npz --protocol random angle_out signer_out \
    --epochs 100 --tag base
"""

from __future__ import annotations

import argparse
import json
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kslx.models.conv_transformer import ConvTransformer, count_params
from kslx.splits import PROTOCOLS, SplitError

ClipMeta = namedtuple("ClipMeta", ["word", "signer", "angle", "take_id"])


def load_clips_meta(npz) -> list[ClipMeta]:
    return [
        ClipMeta(word=int(w), signer=int(s), angle=str(a), take_id=str(t))
        for w, s, a, t in zip(npz["y"], npz["signer"], npz["angle"], npz["take_id"])
    ]


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict:
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.view(-1, 1))
    out = {}
    for k in ks:
        out[f"top{k}"] = correct[:, :k].any(dim=1).float().mean().item()
    return out


def run_protocol(protocol: str, X, y, clips: list[ClipMeta], num_classes: int,
                  device: str, epochs: int, batch_size: int, lr: float,
                  seed: int, protocol_kwargs: dict) -> dict:
    split_fn = PROTOCOLS[protocol]
    train_idx, val_idx = split_fn(clips, **protocol_kwargs)

    x_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx]).long()
    x_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx]).long()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    model = ConvTransformer(feature_dim=X.shape[-1], num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_top1 = 0.0
    best_state = None
    x_val_dev = x_val.to(device)
    y_val_dev = y_val.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[{protocol}] epoch {epoch}: loss is {loss.item()} (NaN/Inf) — "
                    f"입력 피처에 이상치가 있는지 확인할 것 (kslx.normalize)"
                )
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_dev)
            metrics = topk_accuracy(val_logits, y_val_dev, ks=(1, 5))
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1}/{epochs} train_loss={epoch_loss/max(1,n_batches):.4f} "
                  f"val_top1={metrics['top1']:.4f} val_top5={metrics['top5']:.4f}")

    return {
        "protocol": protocol, "n_train": len(train_idx), "n_val": len(val_idx),
        "best_top1": best_top1, "final_top1": metrics["top1"], "final_top5": metrics["top5"],
        "state_dict": best_state, "num_classes": num_classes, "feature_dim": X.shape[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--protocol", nargs="+", default=["random"], choices=list(PROTOCOLS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--n-val-signers", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz = np.load(args.data, allow_pickle=False)
    X, y = npz["X"], npz["y"]
    classes = npz["classes"]
    num_classes = len(classes)
    clips = load_clips_meta(npz)
    print(f"[train] X={X.shape} y_classes={num_classes} device={args.device}")

    dummy = ConvTransformer(feature_dim=X.shape[-1], num_classes=num_classes)
    print(f"[train] model params: {count_params(dummy):,}")

    Path("runs").mkdir(exist_ok=True)
    results = []
    for protocol in args.protocol:
        kwargs = {}
        if protocol in ("random", "take_out", "signer_out"):
            kwargs["seed"] = args.seed
        if protocol in ("random", "take_out"):
            kwargs["val_ratio"] = args.val_ratio
        if protocol == "signer_out":
            kwargs["n_val_signers"] = args.n_val_signers
        print(f"\n=== protocol={protocol} ===")
        t0 = time.time()
        try:
            res = run_protocol(protocol, X, y, clips, num_classes, args.device,
                                args.epochs, args.batch_size, args.lr, args.seed, kwargs)
        except SplitError as e:
            print(f"  [SKIP] {protocol}: {e}")
            results.append({"protocol": protocol, "error": str(e)})
            continue
        elapsed = time.time() - t0
        print(f"  n_train={res['n_train']} n_val={res['n_val']} "
              f"best_top1={res['best_top1']:.4f} final_top5={res['final_top5']:.4f} "
              f"({elapsed:.1f}s)")

        ckpt_path = Path("runs") / f"{protocol}_{args.tag}.pt"
        torch.save({
            "state_dict": res["state_dict"], "num_classes": res["num_classes"],
            "feature_dim": res["feature_dim"], "classes": classes, "protocol": protocol,
            "tag": args.tag,
        }, ckpt_path)
        res.pop("state_dict")
        res["checkpoint"] = str(ckpt_path)
        res["elapsed_sec"] = elapsed
        results.append(res)

    summary_path = Path("runs") / f"{args.tag}_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== 요약 ===")
    print(f"{'protocol':<12}{'top1':>8}{'top5':>8}{'n_train':>10}{'n_val':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['protocol']:<12}{'—':>8}{'—':>8}  ({r['error'][:40]}...)")
        else:
            print(f"{r['protocol']:<12}{r['best_top1']*100:>7.1f}%{r['final_top5']*100:>7.1f}%"
                  f"{r['n_train']:>10}{r['n_val']:>8}")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
