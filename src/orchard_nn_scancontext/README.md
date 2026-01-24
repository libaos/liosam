# orchard_nn_scancontext

ROS1 node: subscribe `sensor_msgs/PointCloud2` → build ScanContext (20×60) → run 2D-CNN → publish:
- `/route_id` (`std_msgs/Int32`)
- `/route_conf` (`std_msgs/Float32`)

This is designed to feed `orchard_corridor` Stage-4 `route_id_gate_node` (gating only; no direct control).

## Run (standalone)
```bash
roslaunch orchard_nn_scancontext nn_scancontext.launch \
  cloud_topic:=/points_raw \
  use_gpu:=true
```

## Run with orchard_corridor Stage 4/5
```bash
roslaunch orchard_corridor debug_stage4.launch \
  use_nn_scancontext:=true \
  nn_cloud_topic:=/points_raw
```

## Model
Default weights:
`models/trajectory_localizer_simple2dcnn_acc97.5.pth`

Common overrides:
- `nn_model_path:=/path/to/weights.pth`
- `nn_model_type:=simple2dcnn|enhanced2dcnn|resnet2d`
- `nn_num_classes:=20`

## Prototype DB (no NN weights)
If your `route_id` labels are defined by splitting a specific rosbag into `K` equal segments by **frame count**, the pretrained NN weights may not match. This alternative builds `K` ScanContext prototypes from that bag and does nearest-neighbor matching (still publishes `/route_id` + `/route_conf`).

### Build DB from a rosbag
```bash
rosrun orchard_nn_scancontext build_sc_route_db.py \
  --bag rosbags/2025-10-29-16-05-00.bag \
  --cloud-topic /points_raw \
  --out output/sc_route_db/2025-10-29-16-05-00_points_raw_K20.npz \
  --num-classes 20 \
  --process-hz 2.0 \
  --write-md
```

### Run (standalone)
```bash
rosrun orchard_nn_scancontext sc_route_id_node.py \
  _cloud_topic:=/points_raw \
  _db_path:=/mysda/w/w/lio_ws/output/sc_route_db/2025-10-29-16-05-00_points_raw_K20.npz \
  _metric:=cosine \
  _temperature:=0.02
```

### Run with orchard_corridor Stage 4/5
```bash
roslaunch orchard_corridor debug_stage4.launch \
  use_sc_route_db:=true \
  sc_cloud_topic:=/points_raw \
  sc_db_path:=/mysda/w/w/lio_ws/output/sc_route_db/2025-10-29-16-05-00_points_raw_K20.npz
```

### Offline evaluate + plot (no ROS needed)
```bash
python3 src/orchard_nn_scancontext/scripts/eval_route_id_on_bag.py \
  --bag rosbags/2025-10-29-16-05-00.bag --cloud-topic /points_raw \
  --method db --db output/sc_route_db/2025-10-29-16-05-00_points_raw_K20.npz \
  --process-hz 2.0 --out-dir output/route_id_eval/sc_db_points_raw

python3 src/orchard_nn_scancontext/scripts/plot_route_id_eval_svg.py \
  --csv output/route_id_eval/sc_db_points_raw/predictions.csv \
  --out output/route_id_eval/sc_db_points_raw/pred_vs_gt.svg
```

## Train a 2D-CNN to match your segment labels
The shipped weights (`trajectory_localizer_simple2dcnn_acc97.5.pth`) are trained for a different label definition. To make NN outputs usable with Stage4 gate on **your** bag-splitting labels, you must train on the same label rule (uniform segments by frame count).

### Train
```bash
python3 src/orchard_nn_scancontext/scripts/train_route_id_cnn_on_bag.py \
  --bag rosbags/2025-10-29-16-05-00.bag --cloud-topic /points_raw \
  --out-dir output/nn_train/route20_simple2dcnn_points_raw \
  --model-type simple2dcnn --num-classes 20 \
  --process-hz 2.0 --epochs 30 --device cuda
```

This writes:
- `output/nn_train/.../model_best.pth` (use as `nn_model_path`)
- `output/nn_train/.../eval/predictions.csv` + `report.md` (offline eval on the same samples)

### Stage4 gate metrics (no ROS needed)
```bash
python3 src/orchard_nn_scancontext/scripts/eval_stage4_gate_on_csv.py \
  --csv output/nn_train/route20_simple2dcnn_points_raw/eval/predictions.csv \
  --out-dir output/nn_train/route20_simple2dcnn_points_raw/eval_gate \
  --conf-th 0.6 --stable-n 3 --allowed-jump 1
```

If you trained with `--split-mode contiguous`, the script also writes `eval_train/` and `eval_val/`.
To simulate Stage4 gate on the **full** sequence but score only held-out validation samples:
```bash
python3 src/orchard_nn_scancontext/scripts/eval_stage4_gate_on_csv.py \
  --csv output/nn_train/route20_simple2dcnn_points_raw/eval/predictions.csv \
  --score-idx-csv output/nn_train/route20_simple2dcnn_points_raw/eval_val/predictions.csv \
  --out-dir output/nn_train/route20_simple2dcnn_points_raw/eval_val_gate \
  --conf-th 0.6 --stable-n 3 --allowed-jump 1
```

Sweep `conf_th` to pick a threshold that is accurate when `valid=true`:
```bash
python3 src/orchard_nn_scancontext/scripts/sweep_stage4_gate_conf_th.py \
  --csv output/nn_train/route20_simple2dcnn_points_raw/eval/predictions.csv \
  --conf-th-min 0.3 --conf-th-max 0.9 --conf-th-step 0.05
```

Sweep on held-out validation samples only:
```bash
python3 src/orchard_nn_scancontext/scripts/sweep_stage4_gate_conf_th.py \
  --csv output/nn_train/route20_simple2dcnn_points_raw/eval/predictions.csv \
  --score-idx-csv output/nn_train/route20_simple2dcnn_points_raw/eval_val/predictions.csv \
  --conf-th-min 0.3 --conf-th-max 0.9 --conf-th-step 0.05
```
