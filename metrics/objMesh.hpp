#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace meshio {

struct Point {
    double x{}, y{}, z{};
};

struct Triangle {
    std::size_t a{}, b{}, c{};
};

struct Mesh {
    std::vector<Point> vertices;
    std::vector<Triangle> triangles;
};

inline Point operator+(Point a, Point b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Point operator-(Point a, Point b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Point operator*(double s, Point a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(Point a, Point b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Point cross(Point a, Point b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm2(Point a) { return dot(a, a); }
inline double norm(Point a) { return std::sqrt(norm2(a)); }

inline std::size_t objIndex(const std::string& token, std::size_t vertexCount) {
    const auto slash = token.find('/');
    const std::string value = token.substr(0, slash);
    if (value.empty()) {
        throw std::runtime_error("OBJ face is missing a vertex index");
    }
    const long long index = std::stoll(value);
    const long long resolved = index > 0 ? index - 1 : static_cast<long long>(vertexCount) + index;
    if (resolved < 0 || static_cast<std::size_t>(resolved) >= vertexCount) {
        throw std::runtime_error("OBJ face contains an out-of-range vertex index");
    }
    return static_cast<std::size_t>(resolved);
}

inline Mesh readObjMesh(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open OBJ: " + path);
    }
    Mesh mesh;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string kind;
        row >> kind;
        if (kind == "v") {
            Point point;
            if (!(row >> point.x >> point.y >> point.z)) {
                throw std::runtime_error("Invalid vertex in OBJ: " + path);
            }
            mesh.vertices.push_back(point);
        } else if (kind == "f") {
            std::vector<std::size_t> face;
            std::string token;
            while (row >> token) {
                if (!token.empty() && token[0] == '#') {
                    break;
                }
                face.push_back(objIndex(token, mesh.vertices.size()));
            }
            if (face.size() < 3) {
                throw std::runtime_error("OBJ face has fewer than three vertices: " + path);
            }
            for (std::size_t i = 1; i + 1 < face.size(); ++i) {
                mesh.triangles.push_back({face[0], face[i], face[i + 1]});
            }
        }
    }
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        throw std::runtime_error("OBJ contains no triangle mesh: " + path);
    }
    return mesh;
}

}  // namespace meshio
