---
title: BallHawk Tactical View
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-4.0
---

# BallHawk: tactical view and offside visualiser

Upload a wide broadcast football frame. The app detects players, referees, goalkeepers and the ball (YOLO11s trained at 1280 px),
splits players into two teams by shirt colour, calibrates the frame onto a 105 × 68 m pitch from 32 detected pitch landmarks,
and draws a top-down minimap with the Law 11 offside line.

An offside visualiser, not VAR: feet are box bottoms, only visible players count, and automatic calibration is about 2.8 m off on average.

Part of BallHawk, a computer-vision course project on why YOLO misses the football on broadcast footage and what fixes it.
Training data: football-players-detection and football-field-detection (Roboflow, CC BY 4.0). Example frames: DFL Deutsche Fußball Liga, Bundesliga Data Shootout.
