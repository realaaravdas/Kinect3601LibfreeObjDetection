#include "geometry.h"
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CameraProjector::CameraProjector(double fov_h, double fov_v, double cam_height, double tilt_angle) {
    this->fov_h_rad = fov_h * M_PI / 180.0;
    this->fov_v_rad = fov_v * M_PI / 180.0;
    this->cam_height = cam_height;
    this->tilt_rad = tilt_angle * M_PI / 180.0;
    this->max_depth = 8.0;
}

void CameraProjector::update_config(double cam_height, double tilt_angle) {
    this->cam_height = cam_height;
    this->tilt_rad = tilt_angle * M_PI / 180.0;
}

Point3D CameraProjector::pixel_to_world(double u, double v, double d, int W, int H, Pose robot_pose) {
    // 1. Intrinsics
    double fx = W / (2.0 * std::tan(this->fov_h_rad / 2.0));
    double fy = H / (2.0 * std::tan(this->fov_v_rad / 2.0));
    double cx = W / 2.0;
    double cy = H / 2.0;

    // 2. Camera Frame (x_c, y_c, z_c)
    double z_c = d;
    double x_c = (u - cx) * z_c / fx;
    double y_c = (v - cy) * z_c / fy;

    // 3. Tilt Rotation (Around X-axis of Opt Frame)
    double c = std::cos(this->tilt_rad);
    double s = std::sin(this->tilt_rad);

    // P_tilted = R_tilt * P_cam
    double x_t = x_c;
    double y_t = y_c * c - z_c * s;
    double z_t = y_c * s + z_c * c;

    // 4. To Robot Frame
    // X_rob = Z_opt_tilted
    // Y_rob = -X_opt_tilted
    // Z_rob = -Y_opt_tilted + height
    double x_r = z_t;
    double y_r = -x_t;
    double z_r = -y_t + this->cam_height;

    // 5. To World Frame
    double cr = std::cos(robot_pose.theta);
    double sr = std::sin(robot_pose.theta);

    double x_w = x_r * cr - y_r * sr + robot_pose.x;
    double y_w = x_r * sr + y_r * cr + robot_pose.y;
    double z_w = z_r; // Assuming robot moves on flat ground Z=0 relative to world

    return {x_w, y_w, z_w};
}

bool CameraProjector::is_in_frustum(double wx, double wy, Pose robot_pose) {
    double dx = wx - robot_pose.x;
    double dy = wy - robot_pose.y;
    double dist = std::sqrt(dx*dx + dy*dy);

    if (dist > this->max_depth) return false;

    double obj_angle = std::atan2(dy, dx);
    double diff = obj_angle - robot_pose.theta;

    // Normalize -PI to PI
    while (diff > M_PI) diff -= 2*M_PI;
    while (diff < -M_PI) diff += 2*M_PI;

    return std::abs(diff) < (this->fov_h_rad / 2.0);
}

std::vector<Pose> CameraProjector::get_frustum_polygon(Pose robot_pose) {
    double alpha = this->fov_h_rad / 2.0;

    Pose p1 = {robot_pose.x, robot_pose.y, 0};

    double lx = robot_pose.x + this->max_depth * std::cos(robot_pose.theta + alpha);
    double ly = robot_pose.y + this->max_depth * std::sin(robot_pose.theta + alpha);
    Pose p2 = {lx, ly, 0};

    double rx = robot_pose.x + this->max_depth * std::cos(robot_pose.theta - alpha);
    double ry = robot_pose.y + this->max_depth * std::sin(robot_pose.theta - alpha);
    Pose p3 = {rx, ry, 0};

    return {p1, p2, p3};
}
