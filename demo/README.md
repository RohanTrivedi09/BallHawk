---
title: BallHawk
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Find a 4-pixel football; tactical view and offside line
---

# BallHawk

Why YOLO misses the football on wide broadcast footage, what fixes it, and how far that carries into
tracking, pitch calibration and an offside visualiser. A Computer Vision project, Adani University.

This Space serves the project website and the live demo from one FastAPI app (`app.py`):
- **Frame:** detections (YOLO11s trained at 1280 px), team colours, automatic pitch calibration from 32
  detected landmarks (or manual calibration from four or more marked ones), a top-down map with space
  control, and the Law 11 offside line.
- **Clip:** the same plus tracking, speeds, distance covered, possession and visible team shape, rendered
  in a background job.

An offside visualiser, not VAR: feet are box bottoms, only visible players count, and automatic
calibration is about 2.8 m off on average. Runs on CPU; a 10-second clip takes a few minutes.

Built from the repository with `tools/make_space.sh`. Training data: football-players-detection and
football-field-detection (Roboflow, CC BY 4.0). Example footage: DFL Deutsche Fußball Liga, Bundesliga Data Shootout.
