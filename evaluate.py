"""Evaluate AT-ADD Track 1/2 predictions with the Codabench metrics."""

import argparse
import csv
import json
import os
from pathlib import Path

LABEL_MAP = {"real": 0, "fake": 1}
TRACK2_TYPES = ("speech", "sound", "singing", "music")


def macro_f1_score(y_true, y_pred):
    """Match sklearn.metrics.f1_score(..., average='macro') for these labels."""
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        raise ValueError("Cannot compute Macro-F1 for an empty input.")

    class_scores = []
    for target_class in classes:
        true_positive = sum(
            truth == target_class and prediction == target_class
            for truth, prediction in zip(y_true, y_pred)
        )
        false_positive = sum(
            truth != target_class and prediction == target_class
            for truth, prediction in zip(y_true, y_pred)
        )
        false_negative = sum(
            truth == target_class and prediction != target_class
            for truth, prediction in zip(y_true, y_pred)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return sum(class_scores) / len(class_scores)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score a progress or full-eval prediction CSV using the official "
            "AT-ADD Codabench metrics."
        )
    )
    parser.add_argument(
        "--track",
        required=True,
        choices=("t1", "t2", "atadd-track1", "atadd-track2"),
        help="Challenge track to evaluate.",
    )
    parser.add_argument(
        "--subset",
        required=True,
        choices=("progress", "eval"),
        help="Use progress.csv or the complete eval.csv labels.",
    )
    parser.add_argument(
        "--prediction_csv",
        "--predict_csv",
        dest="prediction_csv",
        required=True,
        help="Prediction CSV containing name,predict columns.",
    )

    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument(
        "--label_csv",
        help="Explicit path to the progress.csv or eval.csv label file.",
    )
    label_group.add_argument(
        "--label_root",
        help=(
            "AT-ADD dataset root containing T1/label and T2/label. "
            "If omitted, ATADD_LABEL_ROOT is used when set."
        ),
    )

    parser.add_argument(
        "--output_json",
        help=(
            "Result JSON path. Defaults to <prediction directory>/"
            "<track>_<subset>_scores.json."
        ),
    )
    return parser.parse_args()


def normalize_track(track):
    return "t1" if track in ("t1", "atadd-track1") else "t2"


def resolve_label_csv(args, track):
    if args.label_csv:
        label_csv = Path(args.label_csv)
        if not label_csv.is_file():
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        return label_csv

    label_root_value = args.label_root or os.environ.get("ATADD_LABEL_ROOT")
    if not label_root_value:
        raise ValueError(
            "Provide --label_csv or --label_root, or set ATADD_LABEL_ROOT."
        )

    root = Path(label_root_value)
    track_number = "1" if track == "t1" else "2"
    filename = f"{args.subset}.csv"
    candidates = (
        root / f"T{track_number}" / "label" / filename,
        root / f"track{track_number}" / filename,
        root / track / filename,
        root / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n  {searched}")


def require_columns(reader, required, csv_kind):
    fieldnames = set(reader.fieldnames or ())
    if not required.issubset(fieldnames):
        raise ValueError(
            f"{csv_kind} CSV must contain columns {sorted(required)}, "
            f"but got {reader.fieldnames}."
        )


def load_labels(label_csv, track):
    labels = {}
    data_types = {}
    required = {"name", "label"}
    if track == "t2":
        required.add("type")

    with label_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader, required, "Label")
        for row in reader:
            name = row["name"].strip()
            label = row["label"].strip().lower()
            if not name:
                raise ValueError("Empty 'name' found in label CSV.")
            if label not in LABEL_MAP:
                raise ValueError(f"Invalid label '{label}' for file '{name}'.")
            if name in labels:
                raise ValueError(f"Duplicate file '{name}' found in label CSV.")

            labels[name] = label
            if track == "t2":
                data_type = row["type"].strip().lower()
                if data_type not in TRACK2_TYPES:
                    raise ValueError(
                        f"Invalid type '{data_type}' for file '{name}'. "
                        f"Valid types: {list(TRACK2_TYPES)}."
                    )
                data_types[name] = data_type

    if not labels:
        raise ValueError("No samples found in label CSV.")
    return labels, data_types


def load_predictions(prediction_csv):
    predictions = {}

    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader, {"name", "predict"}, "Prediction")

        for row in reader:
            name = row["name"].strip()
            if not name:
                raise ValueError("Empty 'name' found in prediction CSV.")
            if name in predictions:
                raise ValueError(f"Duplicate prediction found for file '{name}'.")

            prediction = row["predict"].strip().lower()
            if prediction not in LABEL_MAP:
                raise ValueError(
                    f"Invalid prediction '{prediction}' for file '{name}'. "
                    "Only 'real' or 'fake' are allowed."
                )

            predictions[name] = prediction

    if not predictions:
        raise ValueError("No samples found in prediction CSV.")
    return predictions


def check_name_match(labels, predictions):
    label_names = set(labels)
    prediction_names = set(predictions)
    missing = label_names - prediction_names
    extra = prediction_names - label_names
    if missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} files. "
            f"Example: {sorted(missing)[:5]}"
        )
    if extra:
        raise ValueError(
            f"Unknown files found in prediction CSV: {len(extra)}. "
            f"Example: {sorted(extra)[:5]}"
        )


def evaluate_track1(labels, predictions):
    names = sorted(labels)
    y_true = [LABEL_MAP[labels[name]] for name in names]
    y_pred = [LABEL_MAP[predictions[name]] for name in names]
    macro_f1 = macro_f1_score(y_true, y_pred) * 100
    return {"macro_f1": round(macro_f1, 2)}


def evaluate_track2(labels, data_types, predictions):
    type_true = {data_type: [] for data_type in TRACK2_TYPES}
    type_pred = {data_type: [] for data_type in TRACK2_TYPES}
    for name in sorted(labels):
        data_type = data_types[name]
        type_true[data_type].append(LABEL_MAP[labels[name]])
        type_pred[data_type].append(LABEL_MAP[predictions[name]])

    raw_scores = {}
    for data_type in TRACK2_TYPES:
        if not type_true[data_type]:
            raise ValueError(f"No samples found for type '{data_type}' in label CSV.")
        raw_scores[data_type] = (
            macro_f1_score(type_true[data_type], type_pred[data_type]) * 100
        )

    mean_type_f1 = sum(raw_scores.values()) / len(TRACK2_TYPES)
    return {
        "macro_f1": round(mean_type_f1, 2),
        "speech_f1": round(raw_scores["speech"], 2),
        "sound_f1": round(raw_scores["sound"], 2),
        "singing_f1": round(raw_scores["singing"], 2),
        "music_f1": round(raw_scores["music"], 2),
    }


def main():
    args = parse_args()
    track = normalize_track(args.track)
    label_csv = resolve_label_csv(args, track)
    prediction_csv = Path(args.prediction_csv)
    if not prediction_csv.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

    labels, data_types = load_labels(label_csv, track)
    predictions = load_predictions(prediction_csv)
    check_name_match(labels, predictions)

    if track == "t1":
        scores = evaluate_track1(labels, predictions)
    else:
        scores = evaluate_track2(labels, data_types, predictions)

    output_json = Path(args.output_json) if args.output_json else (
        prediction_csv.parent / f"{track}_{args.subset}_scores.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(scores, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Track: {track.upper()}")
    print(f"Subset: {args.subset}")
    print(f"Labels: {label_csv}")
    print(f"Predictions: {prediction_csv}")
    print(f"Samples: {len(labels)}")
    for metric, value in scores.items():
        print(f"{metric}: {value:.2f}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
