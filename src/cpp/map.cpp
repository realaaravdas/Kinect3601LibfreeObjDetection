#include "map.h"
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

MapManager::MapManager(CameraProjector* projector) {
    this->projector = projector;
    this->next_id = 0;
    this->merge_threshold = 1.0;
    this->robot_pose = {0,0,0};
}

void MapManager::update_pose(double x_cam, double y_cam, double z_cam, double yaw_cam) {
    // Camera Frame (Start): X-Right, Y-Down, Z-Forward
    // Map Frame: X-Forward, Y-Left
    double map_x = z_cam;
    double map_y = -x_cam;
    double map_theta = -yaw_cam;

    this->robot_pose.x = map_x;
    this->robot_pose.y = map_y;
    // Normalize theta
    map_theta = std::fmod(map_theta + M_PI, 2*M_PI);
    if (map_theta < 0) map_theta += 2*M_PI;
    map_theta -= M_PI;

    this->robot_pose.theta = map_theta;
}

void MapManager::update_map(const std::vector<Detection>& detections,
                            const unsigned short* depth_data, int W, int H) {

    if (!depth_data) return;

    // 1. Decay Confidence
    std::vector<int> in_view_indices;
    for (size_t i = 0; i < objects.size(); ++i) {
        if (projector->is_in_frustum(objects[i].x, objects[i].y, robot_pose)) {
            in_view_indices.push_back(i);
            objects[i].health -= 1; // Decay
        }
    }

    // 2. Process Detections
    for (const auto& det : detections) {
        int x1 = (int)det.x1;
        int y1 = (int)det.y1;
        int x2 = (int)det.x2;
        int y2 = (int)det.y2;

        int cx = (x1 + x2) / 2;
        int cy = (y1 + y2) / 2;

        cx = std::max(0, std::min(cx, W-1));
        cy = std::max(0, std::min(cy, H-1));

        // Get Depth
        std::vector<unsigned short> valid_depths;
        valid_depths.reserve((x2-x1)*(y2-y1));

        // Limit ROI to image
        int rx1 = std::max(0, x1);
        int ry1 = std::max(0, y1);
        int rx2 = std::min(W, x2);
        int ry2 = std::min(H, y2);

        // Optimization: Stride
        int stride = 2;
        for (int y = ry1; y < ry2; y+=stride) {
            for (int x = rx1; x < rx2; x+=stride) {
                unsigned short d = depth_data[y * W + x];
                if (d > 0) valid_depths.push_back(d);
            }
        }

        if (valid_depths.empty()) continue;

        std::nth_element(valid_depths.begin(), valid_depths.begin() + valid_depths.size()/2, valid_depths.end());
        unsigned short d_mm = valid_depths[valid_depths.size()/2];
        double d_m = d_mm / 1000.0;

        if (d_m > projector->max_depth || d_m < 0.3) continue;

        // Project
        Point3D p = projector->pixel_to_world(cx, cy, d_m, W, H, robot_pose);

        // Match
        bool matched = false;
        for (int idx : in_view_indices) {
            MapObject& obj = objects[idx];
            double dist = std::sqrt(std::pow(obj.x - p.x, 2) + std::pow(obj.y - p.y, 2));

            if (dist < merge_threshold && obj.class_id == det.cls) {
                // Match
                double alpha = 0.3;
                obj.x = (1.0 - alpha) * obj.x + alpha * p.x;
                obj.y = (1.0 - alpha) * obj.y + alpha * p.y;
                obj.z = (1.0 - alpha) * obj.z + alpha * p.z;
                obj.confidence = std::max(obj.confidence, (double)det.conf);

                // Heal Logic: Stronger Heal to counter decay
                obj.health = std::min(obj.health + 10, 100);
                matched = true;
                break;
            }
        }

        if (!matched) {
            MapObject new_obj;
            new_obj.id = next_id++;
            new_obj.class_id = det.cls;
            new_obj.x = p.x;
            new_obj.y = p.y;
            new_obj.z = p.z;
            new_obj.confidence = det.conf;
            new_obj.health = 50; // Start Health
            objects.push_back(new_obj);
        }
    }

    // 3. Cleanup
    objects.erase(std::remove_if(objects.begin(), objects.end(),
        [](const MapObject& o){ return o.health <= 0; }), objects.end());
}

std::vector<MapObject> MapManager::get_objects() {
    return objects;
}

Pose MapManager::get_pose() {
    return robot_pose;
}
