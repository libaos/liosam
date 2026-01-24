cmake_minimum_required(VERSION 3.0.2)
project(orchard_nn_scancontext)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/nn_scancontext_node.py
    scripts/sc_route_id_node.py
    scripts/build_sc_route_db.py
    scripts/eval_route_id_on_bag.py
    scripts/plot_route_id_eval_svg.py
    scripts/eval_stage4_gate_on_csv.py
    scripts/sweep_stage4_gate_conf_th.py
    scripts/train_route_id_cnn_on_bag.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
