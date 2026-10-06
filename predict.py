"""Run the current P_IS fold model on inputs matching input_contract.json."""
import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

from sf02_model import CLASS_NAMES, FoldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--route", required=True, choices=("asa", "other"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--body", type=Path)
    parser.add_argument("--lengths", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("output must use the .npz extension")
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if args.route == "asa" and (args.body is None or args.lengths is None):
        parser.error("ASA inputs require --body and --lengths")
    x = sparse.load_npz(args.features)
    model = FoldModel(args.weights, args.fold, args.device)
    if args.route == "asa":
        body = np.load(args.body, mmap_mode="r", allow_pickle=False)
        lengths = np.load(args.lengths, mmap_mode="r", allow_pickle=False)
        probabilities = model.predict_asa(x, body, lengths)
    else:
        probabilities = model.predict_router(x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=probabilities,
            labels=probabilities.argmax(axis=1),
            class_names=np.asarray(CLASS_NAMES),
            fold=np.asarray(args.fold),
        )
    print(f"Saved {len(probabilities)} predictions to {args.output}")


if __name__ == "__main__":
    main()
