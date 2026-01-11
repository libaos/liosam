# orchard_ch5_pipeline

Launch-only package to wire the Chapter 5 pipeline end-to-end:
- RandLA-Net segmentation + row fitting (orchard_row_mapping)
- Optional tree clustering circles (orchard_tree_clusters_node.py)
- ScanContext FSM mode output (/fsm/mode)
- TEB parameter switching based on /fsm/mode
- Optional move_base_benchmark (TEB) and bag_route_replay

## 1) Replay a rosbag (offline reproduction)

```bash
roslaunch orchard_ch5_pipeline ch5_from_bag.launch \
  bag:=/abs/path/to.bag rate:=1
```

## 2) Live run with move_base + TEB

```bash
roslaunch orchard_ch5_pipeline ch5_live_teb.launch \
  map_filename:=/abs/path/to/map.yaml \
  teb_server:=/move_base_benchmark/TebLocalPlannerROS
```

Notes:
- To use your own move_base node, set `enable_move_base:=false` and keep only
  `enable_teb_switcher:=true` (set `teb_server` to your node name).
- To inject tree points into the local costmap, keep
  `enable_tree_cloud_costmap:=true` and adjust
  `config/local_costmap_tree_cloud.yaml` as needed.

