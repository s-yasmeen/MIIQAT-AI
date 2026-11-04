# Hajj and Masjid al-Nabawi Dataset Integration Protocol

## Decision

Use the two sources as complementary, separate evaluation streams. Do not concatenate them into a single multi-zone time series: they come from different sites, collection periods, camera layouts and annotation schemes, and neither source currently supplies a validated shared zone graph.

## Dataset A — HAJJv2 and HAJJv2-CrowdCount

Sources:
- Official HAJJv2 repository: https://github.com/KAU-Smart-Crowd/HAJJv2_dataset
- Crowd-count annotations: https://github.com/reem-8899/HAJJv2-CrowdCount
- Original study: https://www.mdpi.com/2079-9292/12/5/1165
- CrowdCount benchmark preprint: https://arxiv.org/abs/2607.07322

Use:
1. Use the crowd-count CSVs for count-stream evaluation and forecasting baselines.
2. Use the official HAJJv2 behaviour annotations for a separate abnormal-behaviour classification/aggregation task, if the original videos and annotations are obtained under the applicable research terms.
3. Preserve video IDs and timestamps. Split by complete video for visual generalization; for online temporal forecasting, make the within-video evaluation explicit and keep a separate video-held-out result.
4. Report count MAE/RMSE and bias; report behaviour metrics per class (macro-F1, precision, recall, PR-AUC where appropriate). The behaviour labels are imbalanced, so accuracy alone is inadequate.

Limits:
- The CrowdCount repository says it contains annotations, not original videos; it marks its own license as TBC. Confirm reuse/redistribution terms with the authors before copying annotations into this repository or redistributing derived files.
- The original HAJJv2 paper describes research/non-commercial use. Follow the source terms and obtain any required permission.
- The HAJJv2 scenes are distinct locations, not a synchronized network of adjacent zones. Scene labels must not be treated as graph edges.
- A count stream from a single video is not evidence of multi-zone forecasting or intervention effectiveness.

## Dataset B — Masjid al-Nabawi large-gathering image dataset

Sources:
- Dataset landing page: https://adnan-nadeem.com/data-repository/
- Figshare DOI: https://doi.org/10.6084/m9.figshare.19775152.v3
- Dataset paper: https://doi.org/10.32604/cmc.2023.035012

Use:
1. Restrict the MIIQAT work to the **RawDataFrames** full-frame images, subject to permission and terms confirmation.
2. Use aggregate, non-identifying crowd-count/density estimation only. Do not use ProcessedFaceImages, face recognition, identity matching, missing-person tracking, or persistent individual trajectories for this project.
3. Build a dataset inventory first: file count, image dimensions, directory/clip grouping, available camera identifiers, sequence ordering, and any supplied video_summary metadata.
4. If reliable clip/camera/time metadata exists, evaluate image-based crowd counting by clip/camera-held-out splits. If timestamps are absent or only frame order is available, report frame-order experiments and do not claim real-time temporal forecasting.
5. Do not infer physical adjacency, zone capacity, flow direction, or camera synchronization from camera count alone.

Metadata caveat:
- The landing page describes 24 cameras in its source narrative, but its video_summary table reports 14 cameras, 34 clips and 4,613 frames. Resolve this discrepancy from the downloaded metadata and authors before making camera-coverage claims.
- The landing page says the dataset contains 2,000 processed face crops for 250 people, including children, and the site displays an all-rights-reserved notice. Obtain written permission and confirm permitted research use before processing or publishing results. Raw frames may still contain identifiable people; apply data minimization and do not publish identifiable samples.

## MIIQAT integration boundary

Keep three benchmark tracks distinct:

| Track | Input | Valid output | Claims not supported |
|---|---|---|---|
| Hajj count stream | Per-second human count annotations | Count forecasting metrics | Multi-zone graph forecasting; intervention impact |
| Hajj behaviour | HAJJv2 object/behaviour annotations | Per-class anomaly metrics and aggregate indicators | Operational risk ground truth |
| Masjid imagery | Permitted full-frame images and verified metadata | Aggregate crowd-count/density metrics | Connected-zone flows unless camera mapping and synchronized timing are independently verified |

The two visual datasets may support perception modules that produce aggregate observations. They do not, by themselves, calibrate the MIIQAT network simulator or validate intervention outcomes. Any synthetic mapping from a camera/scene to a MIIQAT zone must be labelled as an assumption and tested separately.

## Required local data layout

Do not commit raw videos, full-frame images, face crops, or source annotations unless redistribution permission is explicit.

```text
data/
  external/
    hajjv2/
      crowdcount/
        train_counts.csv
        test_counts.csv
      behaviour/
        train.csv
        test.csv
    masjid_al_nabawi/
      RawDataFrames/
      video_summary/
  processed/
    hajjv2/
    masjid_al_nabawi/
```

Keep external data out of version control. Store source URLs, retrieval date, checksum, terms/permission status and preprocessing version in a local manifest. The repository should contain code, protocols and aggregate benchmark outputs only where the source terms permit.

## Execution sequence

1. Obtain and verify reuse permissions for both sources.
2. Download data locally; do not commit source media.
3. Inspect actual CSV/image schemas and the Masjid video_summary file.
4. Run the existing HAJJv2 count benchmark on the verified annotation CSVs and preserve per-video predictions.
5. Implement a Masjid raw-frame inventory and aggregate counting evaluation only after the image metadata and permitted use are confirmed.
6. Keep the final evaluation video/clip-held-out. Report source-specific metrics and limitations separately.
7. Only after obtaining synchronized zone IDs, timestamps, camera-to-zone mapping and documented adjacency should either source be considered for graph-temporal model input.


## Implementation status — 4 November 2026

- Added `benchmarks/masjid_frame_inventory.py`. It inventories filenames, file sizes, extensions and optional SHA-256 checksums under a user-supplied `RawDataFrames` directory. It does not decode image pixels and excludes directories whose names indicate face/person/identity/subject/processed imagery.
- Hardened `benchmarks/hajj_count_forecast.py` to validate required columns, reject duplicate video timestamps before insertion, and reject negative or non-finite counts.
- Added `tests/test_dataset_tools.py` for duplicate timestamps, invalid counts and exclusion of face-image directories.
- No source dataset files have been copied into this repository. No Masjid image pixels have been inspected or processed.

### Verified source metadata

The public HAJJv2-CrowdCount README describes 18 videos, 9 train / 9 test, 383 manually counted frames at 1 FPS, and a maximum annotated count of 1,800. Its annotation repository currently labels its license and citation as TBC. The source README's stated schema is `video,time_s,GT_count`. The released CSVs use that schema.

The Masjid dataset paper describes 4,613 raw frames from 34 clips across 24 cameras, alongside 2,000 face crops for 250 people. The latter include children. For MIIQAT, use only a permitted full-frame subset for aggregate counts; never use the face-crop collection or person-identification labels. The actual camera/clip mapping and time metadata still need inspection from the source's supplied metadata.

### Release gate before data processing

1. Obtain written confirmation of permitted research use for HAJJv2-CrowdCount annotations (license currently TBC).
2. Verify the exact Figshare item license and dataset terms for the Masjid data; confirm that aggregate crowd counting on RawDataFrames is permitted.
3. Confirm the source metadata's camera-count discrepancy and whether clip timestamps or frame order are recoverable.
4. Only then download the permitted files locally and run the benchmarks. Keep source media and annotations out of Git unless redistribution is expressly permitted.

Until these gates are cleared, the repository contains reproducible tooling and protocol, not claimed empirical results from either new download.
