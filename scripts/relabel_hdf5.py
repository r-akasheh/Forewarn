import argparse
import json
import re
import shutil

import h5py


def load_labels(labels_json_path):
    """Load {episode_id -> label} mapping from an eval trajectories JSON file."""
    with open(labels_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError("Invalid labels JSON: 'episodes' must be a list")

    labels_by_episode_id = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        if "episode_id" not in episode or "label" not in episode:
            continue

        episode_id = int(episode["episode_id"])
        label = int(episode["label"])

        if episode_id in labels_by_episode_id:
            raise ValueError(f"Duplicate episode_id in labels JSON: {episode_id}")

        labels_by_episode_id[episode_id] = label

    if not labels_by_episode_id:
        raise ValueError("No valid (episode_id, label) pairs found in labels JSON")

    return labels_by_episode_id


def _extract_index_from_key(key):
    match = re.match(r"^.+_(\d+)$", key)
    if not match:
        return None
    return int(match.group(1))


def _resolve_episode_container(h5_file):
    """Return the group that contains trajectory/demo groups."""
    if "data" in h5_file and isinstance(h5_file["data"], h5py.Group):
        return h5_file["data"]
    return h5_file


def _find_episode_key(container, episode_id):
    """Find group key for one episode id, supporting demo_* and traj_* names."""
    preferred = [f"demo_{episode_id}", f"traj_{episode_id}"]
    for key in preferred:
        if key in container and isinstance(container[key], h5py.Group):
            return key

    for key in container.keys():
        if not isinstance(container[key], h5py.Group):
            continue
        key_idx = _extract_index_from_key(key)
        if key_idx == episode_id:
            return key

    return None


def ensure_data_layout(hdf5_file_path):
    """Restructure root-level traj/demo groups into /data/demo_<id> layout."""
    moved = 0
    with h5py.File(hdf5_file_path, "r+") as h5_file:
        if "data" in h5_file:
            return moved

        source_keys = []
        for key in list(h5_file.keys()):
            if not isinstance(h5_file[key], h5py.Group):
                continue
            key_idx = _extract_index_from_key(key)
            if key_idx is None:
                continue
            source_keys.append((key, key_idx))

        if not source_keys:
            raise ValueError("No trajectory/demo groups found to restructure")

        h5_file.create_group("data")
        for source_key, key_idx in source_keys:
            target_key = f"demo_{key_idx}"
            target_path = f"data/{target_key}"
            if target_path in h5_file:
                raise ValueError(f"Cannot restructure: target already exists: {target_path}")
            h5_file.move(source_key, target_path)
            moved += 1

    return moved


def apply_labels_to_hdf5(hdf5_file_path, labels_by_episode_id):
    """Update each demo_i attr label using pre-annotated JSON labels."""
    updated = 0
    missing_in_hdf5 = []

    with h5py.File(hdf5_file_path, "r+") as h5_file:
        container = _resolve_episode_container(h5_file)
        for episode_id, label in sorted(labels_by_episode_id.items()):
            episode_key = _find_episode_key(container, episode_id)
            if episode_key is None:
                missing_in_hdf5.append(f"episode_id={episode_id}")
                continue

            container[episode_key].attrs["label"] = label
            updated += 1

    return updated, missing_in_hdf5


def apply_uniform_label_to_all_trajectories(hdf5_file_path, label=1):
    """Set the same label for every trajectory/demo group in the resolved container."""
    updated = 0

    with h5py.File(hdf5_file_path, "r+") as h5_file:
        container = _resolve_episode_container(h5_file)
        for key in sorted(container.keys()):
            if not isinstance(container[key], h5py.Group):
                continue
            if _extract_index_from_key(key) is None:
                continue
            container[key].attrs["label"] = int(label)
            updated += 1

    return updated


def main(
    input_hdf5_path,
    output_hdf5_path,
    labels_json_path=None,
    restructure_to_data=False,
    set_all_labels_to_one=False,
):

    # Keep original input untouched unless output path equals input path.
    if input_hdf5_path != output_hdf5_path:
        shutil.copy2(input_hdf5_path, output_hdf5_path)

    if restructure_to_data:
        moved = ensure_data_layout(output_hdf5_path)
        print(f"Restructured file to /data/demo_* layout. Moved {moved} groups.")

    if set_all_labels_to_one:
        updated = apply_uniform_label_to_all_trajectories(output_hdf5_path, label=1)
        print(f"Updated labels to 1 for {updated} demos/trajectories.")
    else:
        labels_by_episode_id = load_labels(labels_json_path)
        updated, missing_in_hdf5 = apply_labels_to_hdf5(output_hdf5_path, labels_by_episode_id)
        print(f"Updated labels for {updated} demos.")

        if missing_in_hdf5:
            print("Warning: these demos were present in JSON but missing in HDF5:")
            for demo_key in missing_in_hdf5:
                print(f"  - {demo_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the original HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Path to the relabeled HDF5 file")
    parser.add_argument("--labels-json", type=str, help="Path to JSON with episodes[].episode_id and episodes[].label")
    parser.add_argument(
        "--set-all-labels-to-one",
        action="store_true",
        help="Set label=1 for all trajectories/demos and skip JSON label loading",
    )
    parser.add_argument(
        "--restructure-to-data",
        action="store_true",
        help="Move root-level trajectory groups into /data/demo_<id> before relabeling",
    )
    args = parser.parse_args()

    if not args.set_all_labels_to_one and not args.labels_json:
        parser.error("--labels-json is required unless --set-all-labels-to-one is used")

    main(
        args.input,
        args.output,
        args.labels_json,
        args.restructure_to_data,
        args.set_all_labels_to_one,
    )
