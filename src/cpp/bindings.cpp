#include "geometry.h"
#include "map.h"
#include <vector>

extern "C" {
    // CameraProjector
    CameraProjector* Projector_new(double fov_h, double fov_v, double cam_height, double tilt_angle) {
        return new CameraProjector(fov_h, fov_v, cam_height, tilt_angle);
    }

    void Projector_delete(CameraProjector* obj) {
        delete obj;
    }

    void Projector_update_config(CameraProjector* obj, double cam_height, double tilt_angle) {
        obj->update_config(cam_height, tilt_angle);
    }

    void Projector_pixel_to_world(CameraProjector* obj, double u, double v, double d, int W, int H,
                                  double rx, double ry, double rtheta,
                                  double* ox, double* oy, double* oz) {
        Pose p = {rx, ry, rtheta};
        Point3D pt = obj->pixel_to_world(u, v, d, W, H, p);
        *ox = pt.x;
        *oy = pt.y;
        *oz = pt.z;
    }

    bool Projector_is_in_frustum(CameraProjector* obj, double wx, double wy, double rx, double ry, double rtheta) {
        Pose p = {rx, ry, rtheta};
        return obj->is_in_frustum(wx, wy, p);
    }

    // Frustum Polygon: Return array of 6 doubles (3 points * 2 coords)
    void Projector_get_frustum(CameraProjector* obj, double rx, double ry, double rtheta, double* out_arr) {
        Pose p = {rx, ry, rtheta};
        auto poly = obj->get_frustum_polygon(p);
        if (poly.size() >= 3) {
            out_arr[0] = poly[0].x; out_arr[1] = poly[0].y;
            out_arr[2] = poly[1].x; out_arr[3] = poly[1].y;
            out_arr[4] = poly[2].x; out_arr[5] = poly[2].y;
        }
    }

    // MapManager
    MapManager* Map_new(CameraProjector* proj) {
        return new MapManager(proj);
    }

    void Map_delete(MapManager* obj) {
        delete obj;
    }

    void Map_update_pose(MapManager* obj, double x, double y, double z, double yaw) {
        obj->update_pose(x, y, z, yaw);
    }

    // Detections passed as flat array: [x1, y1, x2, y2, cls, conf, ...repeat...]
    // Depth passed as flat uint16 array
    void Map_update_map(MapManager* obj, double* dets, int num_dets,
                        unsigned short* depth_data, int W, int H) {
        std::vector<Detection> detections;
        detections.reserve(num_dets);
        for (int i=0; i<num_dets; ++i) {
            Detection d;
            d.x1 = (float)dets[i*6 + 0];
            d.y1 = (float)dets[i*6 + 1];
            d.x2 = (float)dets[i*6 + 2];
            d.y2 = (float)dets[i*6 + 3];
            d.cls = (int)dets[i*6 + 4];
            d.conf = (float)dets[i*6 + 5];
            detections.push_back(d);
        }
        obj->update_map(detections, depth_data, W, H);
    }

    // Get Objects: Returns count, fills buffer.
    // Buffer layout: [id, cls, x, y, z, conf, health, ...repeat...]
    int Map_get_objects(MapManager* obj, double* out_buffer, int max_count) {
        auto objects = obj->get_objects();
        int count = 0;
        for (const auto& o : objects) {
            if (count >= max_count) break;
            out_buffer[count*7 + 0] = (double)o.id;
            out_buffer[count*7 + 1] = (double)o.class_id;
            out_buffer[count*7 + 2] = o.x;
            out_buffer[count*7 + 3] = o.y;
            out_buffer[count*7 + 4] = o.z;
            out_buffer[count*7 + 5] = o.confidence;
            out_buffer[count*7 + 6] = (double)o.health;
            count++;
        }
        return count;
    }

    void Map_get_pose(MapManager* obj, double* out_arr) {
        Pose p = obj->get_pose();
        out_arr[0] = p.x;
        out_arr[1] = p.y;
        out_arr[2] = p.theta;
    }
}
