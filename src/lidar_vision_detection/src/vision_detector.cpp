#include "lidar_vision_detection/vision_detector.h"

namespace lidar_vision_detection {

VisionDetector::VisionDetector() 
  : it_(nh_),
    tf_listener_(tf_buffer_),
    current_cloud_(new pcl::PointCloud<pcl::PointXYZRGB>) {
}

VisionDetector::~VisionDetector() {
}

bool VisionDetector::initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh) {
    nh_ = nh;
    
    // Initialize visualizer
    visualizer_ = std::make_shared<DetectionVisualizer>(nh);
    
    // Load parameters
    private_nh.param("confidence_threshold", confidence_threshold_, 0.5f);
    private_nh.param("nms_threshold", nms_threshold_, 0.4f);
    private_nh.param("input_width", input_width_, 416);
    private_nh.param("input_height", input_height_, 416);
    private_nh.param<std::string>("model_path", model_path_, "");
    private_nh.param<std::string>("config_path", config_path_, "");
    private_nh.param<std::string>("camera_frame_id", camera_frame_id_, "camera");
    
    // Load class names
    std::string class_names_file;
    private_nh.param<std::string>("class_names_file", class_names_file, "");
    
    if (!class_names_file.empty()) {
        std::ifstream file(class_names_file);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    class_names_.push_back(line);
                }
            }
            file.close();
            ROS_INFO_STREAM("Loaded " << class_names_.size() << " class names from " << class_names_file);
        } else {
            ROS_WARN_STREAM("Failed to open class names file: " << class_names_file);
        }
    }
    
    // Setup ROS communication
    image_sub_ = it_.subscribe("image", 1, &VisionDetector::imageCallback, this);
    pointcloud_sub_ = nh.subscribe("points", 1, &VisionDetector::pointCloudCallback, this);
    detection_pub_ = nh.advertise<lidar_vision_detection::DetectedObjectArray>("detections", 1);
    
    // Load the network
    if (!loadNetwork()) {
        ROS_ERROR("Failed to load neural network");
        return false;
    }
    
    ROS_INFO("Vision detector initialized successfully");
    return true;
}

bool VisionDetector::loadNetwork() {
    if (model_path_.empty() || config_path_.empty()) {
        ROS_ERROR("Model path or config path is empty");
        return false;
    }
    
    try {
        // Load the network
        net_ = cv::dnn::readNet(model_path_, config_path_);
        
        // Get output layer names
        std::vector<std::string> outLayersNames;
        std::vector<int> outLayers = net_.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net_.getLayerNames();
        outLayersNames.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            outLayersNames[i] = layersNames[outLayers[i] - 1];
        }
        output_names_ = outLayersNames;
        
        // Set preferred backend and target
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        ROS_INFO("Neural network loaded successfully");
        return true;
    }
    catch (const cv::Exception& e) {
        ROS_ERROR_STREAM("Error loading neural network: " << e.what());
        return false;
    }
}

void VisionDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
        return;
    }
    
    // Run object detection
    std::vector<Detection> detections = runInference(cv_ptr->image);
    
    // Publish detections
    lidar_vision_detection::DetectedObjectArray detection_msg = createDetectionMessage(detections, msg->header);
    detection_pub_.publish(detection_msg);
    
    // Visualize detections
    if (visualizer_) {
        visualizer_->createMarkerArray(detection_msg);
    }
}

void VisionDetector::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::fromROSMsg(*msg, *current_cloud_);
}

std::vector<Detection> VisionDetector::runInference(const cv::Mat& image) {
    std::vector<Detection> detections;
    
    // Create a blob from the input image
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1/255.0, cv::Size(input_width_, input_height_), 
                          cv::Scalar(0,0,0), true, false);
    
    // Set input to the network
    net_.setInput(blob);
    
    // Forward pass
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, output_names_);
    
    // Process outputs
    postProcessDetections(outputs, image, detections);
    
    return detections;
}

void VisionDetector::postProcessDetections(const std::vector<cv::Mat>& outputs, 
                                         const cv::Mat& image,
                                         std::vector<Detection>& detections) {
    // Lists to store detection results
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // Original image dimensions
    int img_height = image.rows;
    int img_width = image.cols;
    
    // Process all outputs
    for (const auto& output : outputs) {
        // For each detection
        for (int i = 0; i < output.rows; ++i) {
            // Get scores for all classes (starting from index 5)
            // YOLO format: [x, y, w, h, confidence, class_scores...]
            auto scores = output.row(i).colRange(5, output.cols);
            cv::Point class_id_point;
            double confidence;
            
            // Get the max score and its index
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id_point);
            confidence *= output.at<float>(i, 4); // Multiply by objectness score
            
            // Filter by confidence threshold
            if (confidence > confidence_threshold_) {
                int center_x = static_cast<int>(output.at<float>(i, 0) * img_width);
                int center_y = static_cast<int>(output.at<float>(i, 1) * img_height);
                int width = static_cast<int>(output.at<float>(i, 2) * img_width);
                int height = static_cast<int>(output.at<float>(i, 3) * img_height);
                int left = center_x - width / 2;
                int top = center_y - height / 2;
                
                class_ids.push_back(class_id_point.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Apply non-maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indices);
    
    // Create final detections
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        
        // Get class label
        std::string label;
        if (class_ids[idx] < class_names_.size()) {
            label = class_names_[class_ids[idx]];
        } else {
            label = "Class " + std::to_string(class_ids[idx]);
        }
        
        detections.emplace_back(box, label, confidences[idx]);
    }
}

lidar_vision_detection::DetectedObjectArray 
VisionDetector::createDetectionMessage(const std::vector<Detection>& detections, 
                                     const std_msgs::Header& header) {
    lidar_vision_detection::DetectedObjectArray msg;
    msg.header = header;
    
    for (const auto& detection : detections) {
        lidar_vision_detection::DetectedObject obj;
        obj.header = header;
        obj.label = detection.label;
        obj.score = detection.confidence;
        
        // Set 2D polygon (rectangle corners)
        geometry_msgs::Point p;
        
        // Top-left
        p.x = detection.bbox.x;
        p.y = detection.bbox.y;
        obj.polygon.push_back(p);
        
        // Top-right
        p.x = detection.bbox.x + detection.bbox.width;
        p.y = detection.bbox.y;
        obj.polygon.push_back(p);
        
        // Bottom-right
        p.x = detection.bbox.x + detection.bbox.width;
        p.y = detection.bbox.y + detection.bbox.height;
        obj.polygon.push_back(p);
        
        // Bottom-left
        p.x = detection.bbox.x;
        p.y = detection.bbox.y + detection.bbox.height;
        obj.polygon.push_back(p);
        
        // Try to estimate 3D position if point cloud is available
        // This is a simple approach - a more sophisticated one would use the camera calibration
        // and match the detection with the point cloud
        
        // Add to array
        msg.objects.push_back(obj);
    }
    
    return msg;
}

} // namespace lidar_vision_detection 