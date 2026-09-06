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

from kslx.augment import augment_features
from kslx.models.conv_transformer import ConvTransformer, count_params
from kslx.splits import PROTOCOLS, SplitError

ClipMeta = namedtuple("ClipMeta", ["word", "signer", "angle", "take_id"])


def load_dataset(path: Path) -> dict:
    """--data 가 .npz 면 그대로, 디렉토리면 kslx.data.build_dataset 이 만든
    npy-dir(메모리맵 가능)로 연다. X 는 npy-dir 일 때만 진짜 np.memmap 이라
    학습 배치를 뽑을 때만 그만큼만 RAM 에 올라간다 — 3000단어 전체처럼
    npz 로는 램이 부족한 대규모 데이터셋에 필요하다."""
    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        return {
            "X": npz["X"], "y": npz["y"], "signer": npz["signer"], "angle": npz["angle"],
            "take_id": npz["take_id"], "classes": npz["classes"],
            "feature_dim": int(npz["feature_dim"]), "t_out": int(npz["t_out"]),
        }
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    n_valid = meta["n_valid"]
    t_out, feature_dim = meta["t_out"], meta["feature_dim"]
    x_mm = np.memmap(path / "X.raw", dtype=np.float32, mode="r",
                      shape=(n_valid, t_out, feature_dim))
    return {
        "X": x_mm, "y": np.load(path / "y.npy"), "signer": np.load(path / "signer.npy"),
        "angle": np.load(path / "angle.npy"), "take_id": np.load(path / "take_id.npy"),
        "classes": np.load(path / "classes.npy"),
        "feature_dim": meta["feature_dim"], "t_out": meta["t_out"],
    }


def load_clips_meta(data: dict) -> list[ClipMeta]:
    return [
        ClipMeta(word=int(w), signer=int(s), angle=str(a), take_id=str(t))
        for w, s, a, t in zip(data["y"], data["signer"], data["angle"], data["take_id"])
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
                  seed: int, protocol_kwargs: dict, augment: bool = False,
                  row_idx: np.ndarray | None = None) -> dict:
    """row_idx: clips/y 가 X(전체 memmap)의 부분집합만 다룰 때, clips 안에서의
    위치(local) -> X 안에서의 실제 행 번호(global) 매핑. None 이면 clips 가
    이미 X 전체에 대응한다(부분집합 아님)."""
    # local = clips/y(부분집합일 수 있음) 안에서의 위치. global = X(항상 전체
    # memmap) 안에서의 실제 행 번호. row_idx 가 없으면 둘은 같다.
    split_fn = PROTOCOLS[protocol]
    train_idx_local, val_idx_local = split_fn(clips, **protocol_kwargs)
    train_idx_local = np.asarray(train_idx_local)
    val_idx_local = np.asarray(val_idx_local)

    train_idx = row_idx[train_idx_local] if row_idx is not None else train_idx_local
    val_idx = row_idx[val_idx_local] if row_idx is not None else val_idx_local

    # val 은 global 기준으로 정렬해서 들고 있는다(local 도 같은 순서로 맞춤) —
    # X 가 memmap 일 때 뒤섞인 순서로 몇만 개를 한 번에 읽으면(=파일 전체를
    # 무작위로 훑는 것과 같음) 호스트 메모리가 급격히 불어나는 걸 실측했다
    # (3000단어 전체 데이터셋에서 재현돼 학습이 강제종료됨). 정렬하면 순차에
    # 가까운 접근이 된다.
    val_order = np.argsort(val_idx)
    val_idx = val_idx[val_order]
    val_idx_local = val_idx_local[val_order]

    # X 를 통째로 학습셋/검증셋 크기만큼 미리 뽑아 올리지 않는다 — X 가
    # memmap(대규모 데이터셋)이면 그 순간 전부 RAM 에 올라가 버린다. 학습은
    # 배치, 검증은 청크 단위로만 그때그때 memmap 에서 읽는다. y 는 (부분집합일
    # 수 있는) local 인덱스로 조회한다.
    y_train_all = y[train_idx_local]
    y_val_all = y[val_idx_local]
    eval_chunk = max(batch_size, 2048)

    model = ConvTransformer(feature_dim=X.shape[-1], num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def evaluate() -> dict:
        model.eval()
        all_logits = []
        with torch.no_grad():
            for start in range(0, len(val_idx), eval_chunk):
                chunk_idx = val_idx[start:start + eval_chunk]
                xb = torch.from_numpy(np.asarray(X[chunk_idx])).to(device)
                all_logits.append(model(xb).cpu())
        logits = torch.cat(all_logits, dim=0)
        target = torch.from_numpy(y_val_all).long()
        return topk_accuracy(logits, target, ks=(1, 5))

    best_top1 = 0.0
    best_state = None
    aug_rng = np.random.default_rng(seed)
    shuffle_rng = np.random.default_rng(seed)
    n_train = len(train_idx)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        perm = shuffle_rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            batch_pos = perm[start:start + batch_size]
            batch_idx = train_idx[batch_pos]
            # 인덱스 정렬 — memmap 랜덤 접근보다 순차/근접 접근이 훨씬 빠르다
            order = np.argsort(batch_idx)
            xb = torch.from_numpy(np.asarray(X[batch_idx[order]]))
            yb = torch.from_numpy(y_train_all[batch_pos[order]]).long()
            if augment:
                xb = augment_features(xb, aug_rng)
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

        metrics = evaluate()
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
    ap.add_argument("--aug", action="store_true", help="학습 배치에 실시간 증강 적용 (kslx.augment)")
    ap.add_argument("--max-classes", type=int, default=None,
                     help="앞에서부터 이 개수의 클래스만 사용 (memmap 대규모 데이터셋에서 "
                          "한 epoch 동안 실제로 건드리는 바이트 수를 줄여 메모리 압박을 낮춘다 — "
                          "3000단어 전체를 한 번에 학습시키면 이 PC(31GB RAM)에서 매 epoch마다 "
                          "거의 전체 20GB+ 를 훑어 free memory가 급락, 실제로 강제종료된 적이 있다)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.data)
    X, y_full = data["X"], data["y"]
    classes = data["classes"]

    row_idx = None
    if args.max_classes is not None and args.max_classes < len(classes):
        # classes 는 항상 오름차순 word id 라, 앞 K개만 쓰는 건 y < K 와 동치이고
        # 그 값들은 이미 0..K-1 이라 클래스 재매핑이 필요 없다.
        row_idx = np.where(y_full < args.max_classes)[0]
        classes = classes[:args.max_classes]
        print(f"[train] --max-classes {args.max_classes}: {len(row_idx)}/{len(y_full)} 행만 사용")

    y = y_full[row_idx] if row_idx is not None else y_full
    signer = data["signer"][row_idx] if row_idx is not None else data["signer"]
    angle = data["angle"][row_idx] if row_idx is not None else data["angle"]
    take_id = data["take_id"][row_idx] if row_idx is not None else data["take_id"]
    num_classes = len(classes)
    clips = load_clips_meta({"y": y, "signer": signer, "angle": angle, "take_id": take_id})
    print(f"[train] X={X.shape} ({'memmap' if isinstance(X, np.memmap) else 'in-RAM'}) "
          f"y_classes={num_classes} device={args.device}")

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
                                args.epochs, args.batch_size, args.lr, args.seed, kwargs,
                                augment=args.aug, row_idx=row_idx)
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

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
