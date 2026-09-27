#!/usr/bin/env python3
"""Exploratory one-step crowd-count forecasting on HAJJv2-CrowdCount annotations.

This evaluates time-series baselines from prior human counts. It does not run
image models and is not a direct reproduction of YOLO-World, SAM3Count, APGCC,
or the MIIQAT synthetic intervention benchmark.
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

MODELS = ("persistence", "rolling_mean_3", "linear_trend_5", "training_global_mean")


def read_counts(path):
    series = defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            series[row["video"]][int(row["time_s"])] = float(row["GT_count"])
    if not series:
        raise ValueError(f"No annotation rows found in {path}")
    for video, points in series.items():
        times = sorted(points)
        if len(times) != len(set(times)):
            raise ValueError(f"Duplicate timestamps in {video}")
    return series


def forecast(history, train_mean):
    recent = history[-5:]
    slopes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    trend = recent[-1] + statistics.mean(slopes) if slopes else recent[-1]
    return {
        "persistence": history[-1],
        "rolling_mean_3": statistics.mean(history[-3:]),
        "linear_trend_5": max(0.0, trend),
        "training_global_mean": train_mean,
    }


def bootstrap_ci(grouped_errors, seed=2026, repetitions=5000):
    videos = list(grouped_errors)
    if len(videos) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        picked = [rng.choice(videos) for _ in videos]
        numerator = sum(sum(grouped_errors[v]) for v in picked)
        denominator = sum(len(grouped_errors[v]) for v in picked)
        samples.append(numerator / denominator)
    samples.sort()
    return [samples[int(.025 * (repetitions - 1))], samples[int(.975 * (repetitions - 1))]]


def evaluate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video"]].append(row)
    result = {"n_frames": len(rows), "n_videos": len(grouped), "models": {}}
    for model in MODELS:
        errors = defaultdict(list)
        signed = []
        squared = []
        for row in rows:
            residual = row["prediction"][model] - row["ground_truth"]
            errors[row["video"]].append(abs(residual))
            signed.append(residual)
            squared.append(residual * residual)
        absolute = [error for values in errors.values() for error in values]
        result["models"][model] = {
            "mae": statistics.mean(absolute),
            "rmse": math.sqrt(statistics.mean(squared)),
            "bias_prediction_minus_truth": statistics.mean(signed),
            "video_cluster_bootstrap_95pct_mae_ci": bootstrap_ci(errors),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--out-dir", default="hajj_count_forecast_results")
    args = parser.parse_args()
    if args.min_history < 1:
        parser.error("--min-history must be at least 1")

    train = read_counts(args.train_csv)
    test = read_counts(args.test_csv)
    train_mean = statistics.mean(value for series in train.values() for value in series.values())
    rows = []
    for video, series in sorted(test.items()):
        times = sorted(series)
        history = []
        for timestamp in times:
            truth = series[timestamp]
            if len(history) >= args.min_history:
                rows.append({
                    "video": video,
                    "time_s": timestamp,
                    "ground_truth": truth,
                    "prediction": forecast(history, train_mean),
                })
            history.append(truth)

    if not rows:
        raise ValueError("No test rows remain after the warm-up period")
    main_rows = [row for row in rows if row["video"] != "Testing_12"]
    extreme_rows = [row for row in rows if row["video"] == "Testing_12"]
    report = {
        "label": "HAJJv2 real annotation benchmark; temporal baselines only",
        "source": "https://github.com/reem-8899/HAJJv2-CrowdCount",
        "task": "one-step-ahead crowd-count forecasting from prior per-second human counts",
        "warmup_seconds": args.min_history,
        "train_video_count": len(train),
        "test_video_count": len(test),
        "train_annotation_count": sum(len(x) for x in train.values()),
        "test_annotation_count": sum(len(x) for x in test.values()),
        "scoring_rows_after_warmup": len(rows),
        "all_test": evaluate(rows),
        "main_test_excluding_Testing_12": evaluate(main_rows),
        "extreme_Testing_12": evaluate(extreme_rows),
        "limitations": [
            "This is not image-based crowd counting and does not reproduce the published YOLO-World, SAM3Count, or APGCC evaluation.",
            "Forecasts use prior ground-truth counts from the same test video, so this measures online temporal extrapolation with an observed count stream.",
            "The benchmark does not evaluate MIIQAT's risk scoring or intervention policy.",
            "The original HAJJv2 video files are not included in the annotation repository.",
        ],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["video", "time_s", "ground_truth", *MODELS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "video": row["video"],
                "time_s": row["time_s"],
                "ground_truth": row["ground_truth"],
                **row["prediction"],
            })
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
