import argparse
import warnings

import h5py
import numpy as np


def _sorted_demo_keys(data_group):
    keys = list(data_group.keys())
    try:
        return sorted(keys, key=lambda x: int(x.split("_")[-1]))
    except Exception:
        return sorted(keys)


def _resolve_action_key(demo_group, preferred_key=None):
    if preferred_key is not None:
        if preferred_key not in demo_group:
            raise KeyError(f"Requested action key '{preferred_key}' not found in demo group")
        return preferred_key
    if "actions" in demo_group:
        return "actions"
    if "actions_abs" in demo_group:
        return "actions_abs"
    raise KeyError("No action dataset found (expected one of: actions, actions_abs)")


def _build_demo_records(data_group, action_key=None, required_len=None):
    records = []
    for key in _sorted_demo_keys(data_group):
        demo = data_group[key]
        resolved_action_key = _resolve_action_key(demo, action_key)
        seq_len = len(demo[resolved_action_key])
        if required_len is not None and seq_len != required_len:
            continue
        if "label" not in demo.attrs:
            raise KeyError(f"Demo '{key}' missing required attribute 'label'")
        label = int(demo.attrs["label"])
        records.append(
            {
                "key": key,
                "label": label,
                "length": int(seq_len),
            }
        )
    return records


def _stratified_holdout(records, per_label, seed):
    rng = np.random.default_rng(seed)
    by_label = {}
    for rec in records:
        by_label.setdefault(rec["label"], []).append(rec["key"])

    holdout = []
    for label in sorted(by_label.keys()):
        keys = by_label[label]
        n = min(per_label, len(keys))
        if n < per_label:
            warnings.warn(
                f"Label {label} has only {len(keys)} demos; taking {n} for holdout instead of {per_label}."
            )
        if n > 0:
            sampled = rng.choice(keys, size=n, replace=False).tolist()
            holdout.extend(sampled)

    holdout_set = set(holdout)
    train = [rec["key"] for rec in records if rec["key"] not in holdout_set]
    return train, holdout


def _copy_split(src_data_group, out_path, selected_keys):
    with h5py.File(out_path, "w") as out_f:
        # Preserve file-level attrs.
        with h5py.File(src_data_group.file.filename, "r") as src_f:
            for attr_name in src_f.attrs:
                out_f.attrs[attr_name] = src_f.attrs[attr_name]

        out_data_group = out_f.create_group("data")
        for attr_name in src_data_group.attrs:
            out_data_group.attrs[attr_name] = src_data_group.attrs[attr_name]

        for i, key in enumerate(selected_keys):
            src_data_group.file.copy(f"data/{key}", out_data_group, name=f"demo_{i}")
            # Keep original key for traceability after reindexing.
            out_data_group[f"demo_{i}"].attrs["original_demo_key"] = key


def split_hdf5_file(args):
    src_path = args.file_path

    with h5py.File(src_path, "r") as f:
        if "data" not in f:
            raise ValueError("Invalid HDF5 file: missing top-level 'data' group")
        data_group = f["data"]

        records = _build_demo_records(
            data_group,
            action_key=args.action_key,
            required_len=args.required_seq_len,
        )
        if not records:
            raise ValueError("No demos available after applying filters.")

        train_keys, holdout_keys = _stratified_holdout(
            records,
            per_label=args.holdout_per_label,
            seed=args.seed,
        )

        # Keep deterministic order in outputs.
        ordered = [rec["key"] for rec in records]
        train_set = set(train_keys)
        holdout_set = set(holdout_keys)
        train_keys = [k for k in ordered if k in train_set]
        holdout_keys = [k for k in ordered if k in holdout_set]

        _copy_split(data_group, args.train_file, train_keys)
        _copy_split(data_group, args.remaining_file, holdout_keys)

        labels_train = {}
        labels_holdout = {}
        by_key = {rec["key"]: rec["label"] for rec in records}
        for k in train_keys:
            labels_train[by_key[k]] = labels_train.get(by_key[k], 0) + 1
        for k in holdout_keys:
            labels_holdout[by_key[k]] = labels_holdout.get(by_key[k], 0) + 1

        print("Split complete")
        print(f"  Source demos considered: {len(records)}")
        print(f"  Train demos: {len(train_keys)} | labels: {dict(sorted(labels_train.items()))}")
        print(f"  Holdout demos: {len(holdout_keys)} | labels: {dict(sorted(labels_holdout.items()))}")
        print(f"  Train file: {args.train_file}")
        print(f"  Holdout file: {args.remaining_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to input HDF5 file")
    parser.add_argument("--train_file", type=str, required=True, help="Output path for train split")
    parser.add_argument("--remaining_file", type=str, required=True, help="Output path for holdout split")
    parser.add_argument(
        "--action_key",
        type=str,
        default=None,
        choices=["actions", "actions_abs"],
        help="Action dataset name to use; default resolves automatically",
    )
    parser.add_argument(
        "--required_seq_len",
        type=int,
        default=None,
        help="Optional filter: keep only demos with this action sequence length",
    )
    parser.add_argument(
        "--holdout_per_label",
        type=int,
        default=10,
        help="Number of demos to sample into holdout for each label",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling")

    split_hdf5_file(parser.parse_args())
