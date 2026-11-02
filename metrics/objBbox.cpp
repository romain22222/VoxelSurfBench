#include "objMesh.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace meshio;

struct Bounds {
    Point low{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity()};
    Point high{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity()};
};

Bounds boundsOf(const Mesh& mesh) {
    Bounds bounds;
    for (const Point p : mesh.vertices) {
        bounds.low.x = std::min(bounds.low.x, p.x);
        bounds.low.y = std::min(bounds.low.y, p.y);
        bounds.low.z = std::min(bounds.low.z, p.z);
        bounds.high.x = std::max(bounds.high.x, p.x);
        bounds.high.y = std::max(bounds.high.y, p.y);
        bounds.high.z = std::max(bounds.high.z, p.z);
    }
    return bounds;
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: objBbox <pred.obj> <gt.obj>\n";
            return 2;
        }
        const Bounds pred = boundsOf(readObjMesh(argv[1]));
        const Bounds gt = boundsOf(readObjMesh(argv[2]));
        const Point pred_extent = pred.high - pred.low;
        const Point gt_extent = gt.high - gt.low;
        const Point pred_center = 0.5 * (pred.high + pred.low);
        const Point gt_center = 0.5 * (gt.high + gt.low);
        const Point inter_low{std::max(pred.low.x, gt.low.x), std::max(pred.low.y, gt.low.y),
                              std::max(pred.low.z, gt.low.z)};
        const Point inter_high{std::min(pred.high.x, gt.high.x), std::min(pred.high.y, gt.high.y),
                               std::min(pred.high.z, gt.high.z)};
        const Point inter_extent{std::max(0.0, inter_high.x - inter_low.x),
                                 std::max(0.0, inter_high.y - inter_low.y),
                                 std::max(0.0, inter_high.z - inter_low.z)};
        const double inter_volume = inter_extent.x * inter_extent.y * inter_extent.z;
        const double pred_volume = pred_extent.x * pred_extent.y * pred_extent.z;
        const double gt_volume = gt_extent.x * gt_extent.y * gt_extent.z;
        const double union_volume = pred_volume + gt_volume - inter_volume;
        const Point ratio{
            pred_extent.x / std::max(gt_extent.x, 1e-12),
            pred_extent.y / std::max(gt_extent.y, 1e-12),
            pred_extent.z / std::max(gt_extent.z, 1e-12),
        };
        const Point extent_delta = pred_extent - gt_extent;

        std::cout << std::setprecision(17)
                  << "{\n"
                  << "  \"bbox_center_distance\": " << norm(pred_center - gt_center) << ",\n"
                  << "  \"bbox_extent_distance\": " << norm(extent_delta) << ",\n"
                  << "  \"bbox_extent_ratio\": " << std::max({ratio.x, ratio.y, ratio.z}) << ",\n"
                  << "  \"bbox_iou\": " << (union_volume > 0.0 ? inter_volume / union_volume : 0.0) << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "objBbox: " << exc.what() << '\n';
        return 1;
    }
}
