# Ballhawk — Implementation Plan

Computer Vision course project, Adani University. YOLO football detection with a study of why the ball is hard to detect and what fixes it. Tracking and team assignment for the demo.

This document is the source of truth. It reflects real data already measured, not assumptions.

---

## Instructions for whoever picks this up

Read the whole document before writing anything. Sections 3 and 4 record bugs that have already cost time — do not reintroduce them.

- **Training runs on Google Colab (T4 GPU), driven by Claude through the `colab` CLI.** Claude allocates the session (`colab new -s ballhawk --gpu T4`), launches full runs as background processes on the VM, polls the logs, fixes and relaunches on error, downloads results to `results/<run_name>/` locally, and stops the session when work pauses. Ro approves each phase (Section 5) before it starts.
- **Smoke-test before every full run.** 1 epoch, end-to-end through the metrics JSON write. A crash after a full run is the expensive failure (Section 4).
- **One notebook per experiment**, numbered. Every experiment notebook writes a metrics JSON in the exact schema in Section 8, so results can be compared mechanically.
- **Change one variable per experiment.** The whole value of the study is attribution — if two things change at once, the result means nothing.
- **Never report validation numbers as test numbers.** Always pass `split="test"` explicitly.
- **Do not tune anything on the test split.** Choose settings on `val`, report on `test`.
- **Never put the Roboflow API key in a cell or a file.** The current dataset (Section 2) needs no key. If a key is ever needed, it goes through `getpass` only — never through `colab exec`, because the CLI logs all code it sends to `~/.config/colab-cli/history`.
- When a result looks surprising, say so and propose a check. Do not rationalise it.

---

## 1. The project in one paragraph

A standard YOLO model trained on broadcast football footage detects players well and the ball poorly. The dataset profile (Section 2) shows why: the ball is around 3.7 pixels wide when images are fed to the model at 640 px, which is smaller than one cell of YOLO's finest detection grid. The project measures that gap with a vanilla baseline, then tests targeted interventions against it — each one motivated by a specific mechanism, each measured in isolation. The finding is which interventions actually help, by how much, and at what cost in speed.

Training YOLO on this dataset is a common tutorial. The small-object study is what makes this project original. Keep the report framed around it.

---

## 2. Data — profiled

**Source (current):** Hugging Face mirror [`martinjolif/football-player-detection`](https://huggingface.co/datasets/martinjolif/football-player-detection), pinned to revision `e8b8cea002692efd74c945fcdad63e729adc5671`. It is a Roboflow export of a fork of `football-players-detection-3zvbc` (Roboflow workspace `football-project-pifbc`, project `football-players-detection-3zvbc-yyhdl`, version 1), itself derived from the original `roboflow-jvuqo/football-players-detection-3zvbc`. Licence CC BY 4.0 — cite the mirror and both Roboflow projects in the report. Downloads with no API key, so no key handling is needed.

**Why not the original:** the Roboflow download failed for Ro, and no public mirror of the 663-image version profiled in notebook 01 was found. The notebook 01 numbers below are kept for reference only; all study results use the current dataset.

**Splits — re-made by video.** The mirror's own split shares clips across splits (all 19 test clips and all 9 test videos also appear in train), which inflates test metrics. `make_splits.py` assigns every source video (filename prefix before the first `_`) to exactly one split, seed 0, and writes `splits.json`. `ballhawk_common.build_dataset()` lays out the data from it. Whole-video granularity gives 61 / 20 / 19 %, not 70 / 15 / 15.

| Split | Videos | Images | Ball | Goalkeeper | Player | Referee |
|---|---|---|---|---|---|---|
| train | 10 | 226 | 204 | 189 | 4,505 | 534 |
| val | 2 | 74 | 65 | 48 | 1,484 | 164 |
| test | 3 | 72 | 58 | 51 | 1,439 | 165 |

No source frame is shared between splits.

**Native resolution: 1920×1080 for every image.** The gate for interventions A (03) and D (06) passes — the detail exists on disk.

**Measured profile (current dataset, 372 images, 8,906 instances):**

| Class | Instances | Median width at 640 px | 10th–90th percentile | Images containing it |
|---|---|---|---|---|
| player | 7,428 | 8.0 px | 5.3 – 12.7 px | 372 / 372 (100%) |
| referee | 863 | 7.0 px | 4.9 – 14.7 px | 371 / 372 |
| ball | 327 | **3.7 px** | **2.8 – 4.9 px** | 325 / 372 (87%) |
| goalkeeper | 288 | 7.7 px | 6.0 – 10.3 px | 271 / 372 |

Notebook 01 profile of the original 663-image version, for reference: ball 565 instances, median 3.7 px (2.9 – 4.8), in 562 / 663 images; player 8.0 px; referee 7.0 px; goalkeeper 7.7 px. The size distributions match — same source footage.

**What the numbers say:**

1. **Everything is small.** Wide broadcast camera angle. Even players are only 8 px wide at 640. This is a small-object dataset across the board.
2. **The ball is extreme.** Under 4 px at the median, under 5 px at the 90th percentile. YOLO's finest standard head works on a stride-8 grid, so the typical ball is smaller than a single grid cell.
3. **The ball is not rare per frame — it is rare per instance.** It appears in 87% of images, but there is one ball against roughly twenty players per frame. So oversampling ball-containing frames is pointless (there is nothing to oversample). This was in the original plan and has been removed.
4. **The test set holds 58 balls.** Ball metrics will be noisy; small deltas between experiments need the seed-variance check (Section 5) before being called real.

---

## 3. Environment facts (observed, not assumed)

From Ro's actual Colab run:

- **Ultralytics 8.4.152**, Python 3.13, torch 2.11 + CUDA 12.8, Tesla T4 (15 GB).
- Tutorials online mostly target older ultralytics. Defaults have moved. Check behaviour against the installed version, not against blog posts.
- **Default tracker is now `tracktrack.yaml`, not ByteTrack.** Always pass `tracker=` explicitly so the report states which one was used.
- Relevant augmentation defaults in this version: `scale=0.5`, `mosaic=1.0`, `copy_paste=0.0`, `mixup=0.0`, `fliplr=0.5`. See Section 5, intervention B.

Phase 0 findings (checked by Claude on the CLI-allocated T4):

- Latest ultralytics is **8.4.163**; all study runs pin `ultralytics==8.4.163` so every experiment shares one version. Python 3.13, torch 2.11 + CUDA 12.8, Tesla T4 15 GB. albumentations 2.0.8.
- **No `yolo11-p2.yaml` ships.** P2 configs present: `yolov8-p2.yaml`, `yolov8-ghost-p2.yaml`, `yolo26-p2.yaml`. See notebook 05.
- **`copy_paste` is a no-op on this dataset.** `CopyPaste.__call__` returns early when `instances.segments` is empty, and the labels are bbox-only. 04c is dropped.
- **Motion blur is supported without patching:** `model.train(augmentations=[...])` takes a list of Albumentations transforms. A custom list **replaces** the defaults (`Blur`, `MedianBlur`, `ToGray`, `CLAHE` at p=0.01, three more at p=0), so 04b must pass the defaults plus `MotionBlur` — otherwise it changes two things.
- **Class-weighted loss is supported cleanly:** `cls_pw` (0 = off, 1 = full inverse class frequency). The optional class-weighted experiment is viable.
- Pose config for Phase 4: `yolo11-pose.yaml`.
- The CLI only sees runtimes it allocated; a runtime opened in the browser does not appear in `colab sessions`.

---

## 4. Known pitfalls — already hit, already fixed

Both are fixed in notebooks 01 and 02. Carry the fixes into every new notebook.

**Pitfall 1 — `valid` vs `val`.** Roboflow names the folder `valid` on disk, but the `data.yaml` key is `val`. Rewriting yaml paths naively produces a path to a `val/` folder that does not exist, and training fails with `images not found`. Correct pattern:

```python
for key, folder in {"train": "train", "val": "valid", "test": "test"}.items():
    p = pathlib.Path(DATA_DIR) / folder / "images"
    if not p.exists():
        p = pathlib.Path(DATA_DIR) / key / "images"
    if p.exists():
        cfg[key] = str(p)
    else:
        cfg.pop(key, None)
```

**Pitfall 2 — nested save directory.** With `project="runs"`, this ultralytics version saves to `runs/detect/runs/<name>/`, not `runs/<name>/`. Never construct the weights path by hand. Always read it:

```python
SAVE_DIR = pathlib.Path(model.trainer.save_dir)
best = SAVE_DIR / "weights" / "best.pt"
```

**General rule these imply:** a notebook that crashes after a 30-minute training run costs 30 minutes. Any path used after training must be read from the trainer, never assumed.

---

## 5. Phases

Each phase ends with a check-in; the next one starts only on Ro's go-ahead.

| Phase | Notebooks | Status | Content |
|---|---|---|---|
| 0 — Setup and gates | — | Required | T4 session, versions, dataset upload, image-resolution check, P2 config name, `copy_paste` on bbox-only labels, motion blur without patching |
| 1 — Detection study | 02–07, 09 | Required (core finding) | Baseline and interventions below, report assets (Section 9) |
| 2 — Tracking and teams | 08 | Optional | Section 6 |
| 3 — Offside visualiser, single frames | 10–11 | Optional | Section 7 |
| 4 — Automatic pitch calibration | 12 | Optional | Section 7A |
| 5 — Tactical video view and per-frame offside on video | 13 | Optional | Section 7B |

The detection study stays the headline. Phases 2–5 are the demo built on top of it.

### Phase 1 — Experiments

Every experiment compares against the baseline. Model, epochs, seed and everything else stay fixed except the one variable under test.

**Fixed across all runs unless the experiment says otherwise:** `yolo11s.pt`, 100 epochs, batch 16, `seed=0`, `deterministic=True`.

### Notebook 02 — Baseline
Stock settings, imgsz 640. The number every later result is compared against. Must stay vanilla.

### Notebook 03 — Intervention A: training resolution
Train at imgsz 960 and 1280. At 1280 the median ball goes from ~3.7 px to ~7.3 px — roughly one grid cell instead of half of one.
- Two runs. Batch will likely need reducing at 1280 to fit T4 memory — if so, record it; it is a confound and must be stated.
- **Gated on the image-resolution check in Section 2.**
- Expected: largest single gain. Also largest cost in training time and inference FPS. Report both.

### Notebook 04 — Intervention B: ball-aware augmentation
The default `scale=0.5` randomly rescales training images between 0.5× and 1.5×. Applied to a 3.7 px ball, the downscale half pushes it toward 2 px — effectively teaching the model on objects too small to see.
- This is a strong, mechanism-driven intervention and a good report section: the default is actively harmful for this class.
- Three changes, so three runs — one variable each (see Instructions):
  - **04a** — `scale=0.2` only.
  - **04b** — 04a + motion blur. Only via a supported ultralytics hook; no library patching. Method confirmed in Phase 0.
  - **04c** — 04b + `copy_paste` to raise ball instance count. Ultralytics copy-paste relies on segment masks; this dataset is bbox-only. If Phase 0 shows it is a no-op, drop 04c and say so.

### Optional — seed variance
Rerun the baseline with `seed=1`. The test split holds only a few dozen balls, so small deltas may be noise; this gives the noise floor.

### Notebook 05 — Intervention C: P2 detection head
Replaces the removed oversampling experiment. Stock YOLO detects at strides 8, 16, 32. A stride-4 (P2) head gives a grid fine enough for sub-8-pixel objects.
- Use the P2 model variant ultralytics provides. Verify the exact config name in the installed version before writing the notebook — do not guess it.
- **Checked (Section 3): there is no `yolo11-p2.yaml`.** Using `yolov8s-p2` would change the model family as well as the head — two variables. **Decided:** a custom `yolo11s-p2.yaml` (config file, not a library patch) with `yolo11s.pt` weights transferred where shapes match. New P2 layers start untrained — state it in `notes`.
- Most direct architectural answer to the problem. Expect higher memory use and slower inference.

### Notebook 06 — Intervention D: sliced inference (SAHI)
Inference only — no retraining. Tile each image, detect per tile, merge with NMS. Applied on top of the baseline weights.
- **Gated on the image-resolution check.** Useless if images are already 640×640.
- Large multiple on inference time. Report it.

### Notebook 07 — Combined
Stack whatever helped. Report whether gains add up or overlap — they often do not, and noticing that is worth a paragraph.

### Optional — class-weighted loss
Only if the installed ultralytics exposes class weighting cleanly. Do not hand-patch the loss function; a modified library is not a reproducible experiment. If unsupported, drop it and say so.

---

## 6. Phase 2 (optional) — Tracking and team assignment, notebook 08

Build only after the detection study is finished. It is the demo, not the finding.

**Tracking.** `model.track(source=video, tracker="bytetrack.yaml", persist=True)` — tracker named explicitly (Section 3). Do not write a tracker. Measure ID switches per minute and ball track continuity (share of frames with a live ball track). Link continuity back to the detection results: better ball mAP should visibly lengthen ball tracks.

**Team assignment**, unsupervised:
1. Crop each player box, keep the upper portion (jersey only).
2. Mask out pitch-green pixels.
3. Dominant colour in Lab or HSV, not RGB.
4. KMeans, k=2, across all player crops in the clip.
5. Exclude referee and goalkeeper by predicted class.
6. Assign team per track by majority vote across frames — per-frame assignment flickers.

**Ball trail.** Draw the last ~30 ball positions, fading. Makes detection dropouts visible to the eye.

**Output:** annotated MP4, players coloured by team, ball with trail.

**Video source — resolved.** "Match de football France-Allemagne - 16 octobre 2018 - Phase de jeu (1)", by Like tears in rain, Wikimedia Commons, **CC BY-SA 4.0** (1920×1080, 30 fps, 17 s). The annotated output is a derivative: share it under CC BY-SA 4.0 with attribution. It is filmed from the stands at a low angle, not from the broadcast camera — a domain shift to state in the report. Rejected: Latvia–Gibraltar 2026 (CC0, but flares and crowd, no play visible); DFL Bundesliga Data Shootout clips (same domain as the training data, but the Kaggle licence could not be verified — usable only if Ro confirms the terms).

**Ground truth caveat:** the clip has no identity labels, so true ID switches cannot be counted. Notebook 08 reports fragmentation proxies instead: player track IDs per minute and median track length.

---

## 7. Phase 3 (optional) — Offside visualiser on single frames, notebooks 10–11

Build only after the detection study (Section 5) is complete and notebook 08 works. This stage depends on player detection, team assignment and the goalkeeper class all being reliable.

**Deliverable:** given a frame, detect players, assign teams, compute a homography onto a flat pitch model, find the second-last defender, and draw the offside line across the pitch with attackers beyond it flagged. Rendered two ways — overlaid on the broadcast frame, and as a 2D top-down minimap beside it.

**Call it an offside *visualiser*, never an automated VAR system.** Real VAR uses multiple calibrated cameras and limb-level skeletal tracking, because offside is judged on the foremost legal body part. This uses the bottom edge of a bounding box as a foot position. State that limitation in the report abstract, not buried in a footnote — owning it is what makes the work credible.

### Notebook 10 — Homography

Map image pixels to pitch coordinates in metres. Two paths, both built:

**Primary — manual 4-point correspondence.** Ro clicks four known pitch landmarks in the frame (penalty box corners are the easiest to identify unambiguously), and the corresponding real-world coordinates come from official pitch dimensions. `cv2.findHomography` does the rest. Robust, honest, and consistent with the semi-automatic scope.

**Secondary — automatic line detection.** White-line mask by colour threshold, morphological cleanup, then a probabilistic Hough transform, then classify which detected lines are which pitch features. This is genuinely hard on tight shots where few lines are visible. Build it, measure how often it succeeds, and report where it fails. A documented failure is a result; skipping the attempt is a gap.

**Pitch model:** use official dimensions — penalty area 40.32 m × 16.5 m, goal area 18.32 m × 5.5 m, centre circle radius 9.15 m. Pitch length and width vary by stadium, so anchor the homography on penalty-area geometry, which is fixed by the laws of the game.

**Validation — this is the part that makes it a CV project rather than a demo.** There is no offside ground truth available, so evaluate the homography instead:

- **Reprojection error.** Click six or more landmarks, fit on four, measure error on the held-out points. Report in pixels and in metres.
- **Reconstructed dimension check.** Measure a known distance through the homography and compare against its official value. Error in centimetres.
- **Sensitivity analysis.** Perturb each clicked point by 1–5 px and measure how far the offside line moves at the far side of the pitch. Report the centimetres-of-line-shift per pixel-of-click-error.

That sensitivity number is the strongest thing in this stage. It quantifies exactly why real VAR needs calibrated multi-camera rigs, using your own measurements rather than a citation.

### Notebook 11 — Offside logic and rendering

1. **Attacking direction** — infer from the goalkeeper. The dataset has a goalkeeper class, and a keeper sits near the goal being defended. Use it rather than asking the user; it is a nice application of a class the detector already provides.
2. **Foot position** — bottom-centre of each player box, projected through the homography into pitch coordinates. Document the approximation.
3. **Offside line** — among the defending team including the keeper, sort by distance to their own goal line; take the second-last. The line sits at whichever is **nearer** to the goal line: the second-last defender, or the ball. *(Corrected 2026-09-26: this previously said "further from goal", which contradicts Law 11 — a player is offside only if nearer to the opponents' goal line than **both** the ball and the second-last opponent, so the binding one is the nearer of the two. The attacker must also be in the opponents' half.)*
4. **Flag attackers** beyond that line.
5. **Render** — build the line in pitch coordinates spanning the full width, project back to image space with the inverse homography, draw as a translucent band. Draw the 2D minimap alongside, players as coloured dots.

Per-frame only. No temporal logic, no pass-moment detection.

### Risks

- Tight camera shots may not contain four identifiable landmarks. Pick test frames with a visible penalty area.
- Players occluding each other produce merged or missing boxes, which moves the second-last defender. Show one such failure case.
- Bounding-box feet are wrong for a running player. Show that too.

---

## 7A. Phase 4 (optional) — Automatic pitch calibration, notebook 12

The manual 4-point homography (Section 7) works per frame, and the broadcast camera pans, so it cannot drive a video. This phase makes the homography automatic, so the tactical view survives camera movement.

- **Model:** a YOLO pose/keypoint model trained to find fixed pitch landmarks (penalty-box corners, centre-line ends, circle intersections). Verify the pose model name and keypoint-yaml format in the installed ultralytics before writing the notebook.
- **Data — resolved:** Hugging Face `martinjolif/football-pitch-detection` (revision `73488ede6158cbfe05aa7a8e471fc5041eceba8b`), **CC BY 4.0**, a Roboflow export of `football-project-pifbc/football-field-detection-f07vi-d0ele` v1. 317 images (255 / 34 / 28), YOLO pose format, `kpt_shape: [32, 3]` with `flip_idx`. SoccerNet keypoints on HF carry no licence — not used.
- **Keypoint order** follows `SoccerPitchConfiguration.vertices` in roboflow/sports (MIT). **Its dimensions are not official** (penalty box 41.0 × 20.15 m; pitch 120 × 70 m). Use the order only; compute real-world coordinates from the official values in Section 7 (penalty area 40.32 × 16.5 m, goal area 18.32 × 5.5 m, penalty spot 11 m, circle 9.15 m) and a 105 × 68 m pitch for points that depend on pitch size (Bundesliga standard; state the assumption). Verify the index-to-landmark mapping visually on a few labelled images before training.
- **Per frame:** detect keypoints above a confidence threshold, then `cv2.findHomography(..., cv2.RANSAC)`. Fewer than 4 confident keypoints → frame marked as uncalibrated, never guessed.
- **Validation**, using the manual clicks from Section 7 as reference:
  - Reprojection error on the manually clicked frames, pixels and metres.
  - Calibration success rate per frame across a clip.
  - Frame-to-frame homography jitter, with and without temporal smoothing (e.g. exponential moving average of the projected pitch corners). Report both.

## 7B. Phase 5 (optional) — Tactical video view and per-frame offside on video, notebook 13

Combines Phase 2 tracks with Phase 4 homographies.

- **Output:** side-by-side MP4 — annotated broadcast frame, plus a live 2D top-down minimap with team-coloured player dots and the ball trail.
- **Offside on video:** the Section 7 logic run on every calibrated frame. Still per-frame: no pass-moment detection, no temporal offside logic.
- **Optional extras from pitch coordinates:** distance covered and speed per tracked player; team position heatmaps. Speeds depend on homography quality — report them with the Phase 4 jitter figure.
- **Report:** share of frames where the offside line flickers because of detection or calibration errors. It is an honest limitation figure, not something to hide.

---

## 8. Metrics JSON — fixed schema

Every experiment notebook writes exactly this, so notebook 09 can build tables without special cases:

```json
{
  "run_name": "baseline_yolo11s_640",
  "notebook": "02",
  "variable_changed": "none",
  "model": "yolo11s.pt",
  "imgsz": 640,
  "epochs": 100,
  "batch": 16,
  "split": "test",
  "overall": { "mAP50": 0.0, "mAP50_95": 0.0 },
  "per_class": [
    { "class": "ball", "precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0 }
  ],
  "inference_fps": 0.0,
  "train_minutes": 0.0,
  "notes": ""
}
```

`variable_changed` and `notes` are not optional. Record any forced deviation — a reduced batch size at 1280 belongs in `notes`.

---

## 9. Notebook 09 — report assets

Reads every metrics JSON. Produces:
- Main results table: baseline versus each intervention versus combined, all metrics.
- Ball mAP50-95 bar chart across experiments — the headline figure.
- Accuracy versus inference FPS scatter — every intervention's cost, in one picture.
- Failure gallery: frames where the ball is missed, with a one-line cause each (occluded by a leg, against a bright stand, motion blur). Do not skip this.

Figure 1 already exists from notebook 01.

---

## 10. Syllabus coverage

Strong: Unit 2 (object detection, feature learning, video understanding, model fitting via RANSAC in the homography), Unit 4 (datasets and benchmarks, libraries), Unit 5 (tracking, video processing).

Phase 3 closes the Unit 1 gap properly rather than rhetorically — homography *is* the syllabus topic "geometric primitives and transformations", and the automatic line-detection path is edge detection and feature extraction applied for real.

Cheap to strengthen in the write-up, no extra runs:
- Frame Intervention B as Unit 1 material — motion blur is a convolution, scale jitter is a geometric transformation, resolution change is resampling.
- Note that the tracker's motion model is a Kalman filter predicting constant velocity — connects to Units 3 and 5.

---

## 11. Status

| Notebook | State |
|---|---|
| 01 — setup and profile | ✅ Done on the original dataset. Re-profiled on the current dataset in Phase 0 (Section 2) |
| Phase 0 — setup and gates | ✅ Done. Native 1920×1080 → 03 and 06 go ahead. Video-grouped split. Library facts in Section 3 |
| 02 — baseline | ✅ Test: ball mAP50-95 0.110, all 0.509. Seed-1 twin (02b): ball 0.115, all 0.501 → ball noise ≈ 0.005 |
| 06 — SAHI | ✅ 960 px slices (chosen on val): ball mAP50-95 0.113 → 0.188 (same evaluator), 53 → 5.7 FPS |
| 03 — resolution | ✅ Test ball mAP50-95: 960 → 0.242, 1280 → 0.298 (batch 8, confound noted). Val: 0.321 / 0.390. Largest gain |
| 04a — scale 0.2 | ✅ Test 0.130 (+0.020); val +0.016, **within 2× val seed noise (0.028)** → suggestive, not established |
| 04b — + motion blur | ✅ No ball gain vs 04a (test −0.003, val −0.009); all-class test mAP drops 0.502 → 0.475 |
| 05 — P2 head | ⚠️ Worse: test 0.045, val 0.065. Learning curve: slow start (val 0.076 at epoch 10 vs 0.400) and lower plateau. Control 05c (stock head, layers 17–23 re-initialised) separates lost pretraining from the head itself |
| 05b — cls_pw 0.5 | ✅ Test 0.127 (+0.017), val +0.000 → no effect established. Goalkeepers get the largest weight, not the ball |
| Seed replication (Kaggle 2×T4) | ✅ Baseline, 04a, 05b × seeds 0/1/2, all on one platform (44.6 min). Ball mAP50-95 mean ± sd — val: baseline 0.122 ± 0.014, scale 0.2 0.130 ± 0.008, cls_pw 0.5 0.122 ± 0.009; test: 0.111 ± 0.009, 0.118 ± 0.018, 0.118 ± 0.003. **Neither gain is established** (≤ 1.4 standard errors); the +0.02 single-seed gains shrink to ≈ +0.007. True seed sd ≈ 0.009–0.014, about twice the single-pair estimate. Resolution (+0.19) is > 10× this noise. Same seed differs by up to 0.011 across Colab/Kaggle (GPU nondeterminism across stacks) |
| FPS | ✅ Re-timed on an idle VM, median of 3: 640 models 44–47, P2 43.8, 960 43.0, 1280 31.7, SAHI 5.2. Batch-1 latency is overhead-bound |
| 05c — control for 05 | ✅ Stock head, layers 17–23 re-initialised: test ball 0.060, all 0.435 ≈ P2 (0.045 / 0.438). **P2's deficit is lost pretraining, not the stride-4 head**; vs its matched control the P2 head changes nothing measurable. (First attempt invalid: re-init also reset the frozen DFL projection → mAP 0; fixed with an assertion) |
| 07 — combined | ✅ Val rule (margin 2× val seed noise = 0.028) keeps only imgsz 1280 → combined = 03-1280, no retraining. SAHI on the 1280 model *hurts* on val (full frame 0.381, SAHI 960 0.366, SAHI 640 0.258): resolution and SAHI fix the same thing, gains do not stack |
| 09 — report assets | ✅ `results/report/`: results table, ball mAP bar chart, accuracy-vs-FPS scatter, failure gallery (16 misses with causes; several are isolated, clearly visible balls lost only to downscaling) |
| 08 — tracking + teams | ✅ France 2018 clip, ByteTrack. Ball-track continuity 16% (02) → 33% (03-1280). Team vote agreement 97%. **Heavy fragmentation** (261–448 player IDs in 17 s). Camera-shake hypothesis mostly refuted: BoT-SORT with camera-motion compensation cuts IDs only 5–14%. Remaining suspect: per-frame detection dropout on a domain-shifted clip; needs identity ground truth to confirm |
| 10 — homography (manual) | ✅ Ro waived the clicking (2026-09-27); correspondences are the pitch dataset's human keypoint annotations, so 20 test frames (all 4 box corners labelled; 30 midfield frames skipped) instead of 3. Fit on the 4 penalty-box corners: held-out error **0.39 m** median (3.7 px), p90 0.69 m; pitch-size-dependent points 0.82 m; goal-area width error 13 cm median. **Click sensitivity: offside line at the far touchline moves ≈ 10 cm per pixel of click error** (box front; ≈ 12 cm/px 30 m out); 5 px → 0.5–0.6 m mean, 0.8–1.2 m p95. Human-label noise alone (≈ 3.7 px) implies ≈ 35–40 cm of line uncertainty vs 1.17 m for automatic calibration. Notebook 11's single-frame offside with manual calibration is the reference-line path of notebook 14 |
| 12 — pitch calibration | ✅ YOLO11s-pose, 960, 150 epochs (26.7 min). Test (videos never trained on): calibrated 100% of frames, pose mAP50 0.85 / mAP50-95 0.29, median keypoint error 27.8 px, **median reprojection error 2.8 m** (p90 4.7 m) — fine for a minimap, far too coarse for offside. Video: calibrated 96.7% of frames; jitter 4.2 m raw → 1.3 m EMA-smoothed. Found and fixed: roboflow/sports defines keypoints 10/11/18/19 at goal-area y, but the annotators placed them where the penalty arc meets the box front (1.6–2.1 m residual → 0.2–0.3 m after the fix). Pitch data re-split by video with the Phase 3 click-frame videos forced into test (two click frames were inside the original dataset). Next lever: choose KP_CONF on val |
| 13 — tactical view + offside | ✅ Pipeline runs end-to-end (`results/tactical/13_tactical_offside.mp4`, CPU runtime — T4 refused after a full day's quota). Line drawn in 91% of frames but **median frame-to-frame jump 2.1 m, 62% of steps > 1 m**. Minimap geometry visibly wrong on this clip (players spread to x ≈ 95 m and compressed against the far touchline while the camera sees the left half): the keypoint model, trained on the high broadcast camera, does not transfer to low-angle phone footage. Offside *logic* is unit-tested; the *calibration* feeding it is not trustworthy here. Report as a limitation. Better demo: in-domain footage (DFL clips, licence unverified) or test-split stills, where error is measured |
| 14 — in-domain offside, measured | ✅ 16 broadcast frames in both datasets (all in the pitch test split). Keypoint conf chosen on val: 0.5 (higher only loses frames: 0.7 → 69 % calibrated). **Offside line error from automatic calibration: median 1.17 m, p90 3.9 m**; player position error 2.1 m; 1/16 frames flip a call. Same roles for both lines, so this isolates calibration. **Role inference is the weak link:** 'defending team = team nearer the keeper's goal' gives implausible lines on stills (e.g. 9 attackers flagged at x = 63 m) — report as a limitation. Found and fixed: fixed-hue pitch-green masking deleted lime kits (team split 108/27 with dropouts → 120/120, 100 % agreement) — now masked against each frame's grass colour |
| 12b — keypoints at 1280 | ✅ Worse than 960 and confounded: batch auto-reduced 16 → 4 on the T4 (noisy BatchNorm), and the run was resumed at epoch 141 after a kernel restart killed it. Test: pose mAP50-95 0.168 vs 0.292, reprojection 3.2 m (p90 41.8 m) vs 2.8 m. **Chosen on val: the 960 model** (2.66 m vs 3.73 m, `results/pitch_kp_choice.json`). A fair 1280 test needs a larger GPU. **Matched-batch control (Kaggle 2×T4, 2026-09-28):** both sizes at batch 4, fresh. Val median reprojection: 1280 **2.94 m** vs 960 4.03 m (pose mAP50-95 0.263 vs 0.221) — resolution helps at equal batch. Test is mixed (median 3.50 vs 3.18 m; both have p90 tails of 10–40 m), so likely but not proven on 50 frames. Batch size matters more: the 960 batch-16 model (val 2.66 m) beats both and stays in the pipeline, chosen on val. A 1280 batch-16 run still needs a 24 GB GPU (Modal / Lightning require a payment method). |
| 08 / 13 reruns | ✅ Team-colour fix applied; France clip team agreement 96.7 % → 97.3 % |
| 08b — tracking, DFL clip | ✅ In domain: DFL clip 121364_9 (test match, never trained on; licence unverified → metrics and stills only, video not published). Ball-track continuity 10 % → 25 % (640 → 1280 detector). 101–130 player IDs in 30 s (phone clip: 261–448 in 17 s). Team agreement 97.3 % |
| 13b — tactical + offside, DFL clip | ✅ In domain: calibrated 100 % of 750 frames, median frame-to-frame line jump **0.085 m** (phone clip 2.1 m), 11.8 % of steps > 1 m (phone clip 62 %). Minimap geometry and team colours correct on inspection. The phone-clip failure was domain shift |

---

## Out of scope

Writing a detector or tracker from scratch. Re-identification across camera cuts. Limb-level or skeletal offside precision. Multi-camera calibration. Automatic detection of the moment the ball is struck. Temporal offside logic across frames. Real-time inference on Ro's Mac.

Per-frame offside on video is **not** out of scope any more — it is optional Phase 5 (Section 7B), still without temporal logic.
