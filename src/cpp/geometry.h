#pragma once
#include <vector>
#include <cmath>

struct Point3D {
    double x, y, z;
};

struct Pose {
    double x, y, theta;
};

class CameraProjector {
public:
    CameraProjector(double fov_h, double fov_v, double cam_height, double tilt_angle);
    void update_config(double cam_height, double tilt_angle);
    Point3D pixel_to_world(double u, double v, double d, int W, int H, Pose robot_pose);
    bool is_in_frustum(double wx, double wy, Pose robot_pose);
    std::vector<Pose> get_frustum_polygon(Pose robot_pose);

    double max_depth;

private:
    double fov_h_rad, fov_v_rad;
    double cam_height;
    double tilt_rad;
};
