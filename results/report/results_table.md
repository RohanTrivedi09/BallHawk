| notebook | run_name | variable_changed | imgsz | batch | mAP50 | mAP50_95 | ball_P | ball_R | ball_mAP50 | ball_mAP50_95 | fps | train_min | label | colour |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 02 | baseline_yolo11s_640 | none | 640 | 16 | 0.746 | 0.509 | 0.739 | 0.31 | 0.305 | 0.11 | 46.108 | 10.303 | 02 baseline | #8f8e89 |
| 02b | baseline_yolo11s_640_seed1 | seed 0 -> 1 | 640 | 16 | 0.749 | 0.501 | 0.864 | 0.293 | 0.314 | 0.115 | 46.992 | 11.174 | 02b baseline, seed 1 | #8f8e89 |
| 03 | res_yolo11s_1280 | imgsz 640 -> 1280 | 1280 | 8 | 0.845 | 0.629 | 0.839 | 0.54 | 0.566 | 0.298 | 31.682 | 30.427 | 03 imgsz 1280 (batch 8) | #2a78d6 |
| 03 | res_yolo11s_960 | imgsz 640 -> 960 | 960 | 16 | 0.825 | 0.596 | 0.954 | 0.5 | 0.532 | 0.242 | 43.036 | 18.699 | 03 imgsz 960 | #2a78d6 |
| 04a | aug_scale02_yolo11s_640 | scale 0.5 -> 0.2 | 640 | 16 | 0.752 | 0.502 | 0.891 | 0.328 | 0.335 | 0.13 | 45.951 | 10.964 | 04a scale 0.2 | #2a78d6 |
| 04b | aug_scale02_mblur_yolo11s_640 | added MotionBlur(blur_limit 3-7, p 0.3) on top of 04a | 640 | 16 | 0.729 | 0.475 | 0.862 | 0.322 | 0.33 | 0.127 | 43.775 | 11.084 | 04b 04a + motion blur | #2a78d6 |
| 05 | p2_yolo11s_640 | detection outputs P3-P5 -> P2-P5 | 640 | 16 | 0.681 | 0.438 | 1.0 | 0.0 | 0.183 | 0.045 | 43.802 | 13.495 | 05 P2 head | #2a78d6 |
| 05b | clsw05_yolo11s_640 | cls_pw 0 -> 0.5 | 640 | 16 | 0.753 | 0.505 | 0.903 | 0.32 | 0.361 | 0.127 | 44.434 | 11.334 | 05b class weights | #2a78d6 |
| 05c | ctrl_reinit17_yolo11s_640 | layers 17-23 re-initialised (control for 05) | 640 | 16 | 0.694 | 0.435 | 0.76 | 0.219 | 0.23 | 0.06 | 48.579 | 11.443 | 05c control: head re-init | #8f8e89 |
| 06 | sahi_baseline_s960 | sliced inference: 960px native slices, overlap 0.2, + full-frame pass, NMS merge | 640 | 16 | 0.792 | 0.547 | 0.707 | 0.414 | 0.437 | 0.188 | 5.191 | 0.0 | 06 SAHI 960 on 02 | #2a78d6 |
