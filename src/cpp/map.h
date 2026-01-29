#pragma once
#include <vector>
#include "geometry.h"

struct MapObject {
    int id;
    int class_id;
    double x, y, z;
    double confidence;
    int health;
};

struct Detection {
    float x1, y1, x2, y2;
    int cls;
    float conf;
};

class MapManager {
public:
    MapManager(CameraProjector* projector);
    void update_pose(double x, double y, double z, double yaw);
    void update_map(const std::vector<Detection>& detections,
                   const unsigned short* depth_data, int W, int H);

    std::vector<MapObject> get_objects();
    Pose get_pose();

private:
    CameraProjector* projector;
    Pose robot_pose;
    std::vector<MapObject> objects;
    int next_id;
    double merge_threshold;
};
