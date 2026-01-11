std_msgs/Header header
string label
float32 score
geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions  # 3D bounding box in meters (x, y, z)
geometry_msgs/Point[] polygon     # 2D detection box in image coordinates 