#include "objMesh.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

using namespace meshio;

struct DisjointSet {
    explicit DisjointSet(std::size_t n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    std::size_t find(std::size_t x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    void unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) ++rank[a];
    }
    std::vector<std::size_t> parent;
    std::vector<unsigned char> rank;
};

struct EdgeUse {
    std::size_t count = 0;
    int orientation = 0;
    std::size_t first_face = 0;
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: objTopology <mesh.obj>\n";
            return 2;
        }
        const Mesh mesh = readObjMesh(argv[1]);
        std::map<std::pair<std::size_t, std::size_t>, EdgeUse> edges;
        DisjointSet components(mesh.triangles.size());
        double area = 0.0;
        double signed_volume = 0.0;

        for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
            const auto& t = mesh.triangles[i];
            const Point a = mesh.vertices[t.a];
            const Point b = mesh.vertices[t.b];
            const Point c = mesh.vertices[t.c];
            area += 0.5 * norm(cross(b - a, c - a));
            signed_volume += dot(a, cross(b, c)) / 6.0;
            const std::pair<std::size_t, std::size_t> directed[] = {
                {t.a, t.b}, {t.b, t.c}, {t.c, t.a},
            };
            for (const auto& edge : directed) {
                const auto key = std::minmax(edge.first, edge.second);
                auto& use = edges[key];
                if (use.count == 0) use.first_face = i;
                else components.unite(i, use.first_face);
                ++use.count;
                use.orientation += edge.first < edge.second ? 1 : -1;
            }
        }

        bool watertight = true;
        bool winding_consistent = true;
        for (const auto& [edge, use] : edges) {
            (void)edge;
            if (use.count != 2) watertight = false;
            if (use.count > 2 || (use.count == 2 && use.orientation != 0)) {
                winding_consistent = false;
            }
        }
        std::size_t component_count = 0;
        for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
            if (components.find(i) == i) ++component_count;
        }
        const long long euler = static_cast<long long>(mesh.vertices.size())
            - static_cast<long long>(edges.size())
            + static_cast<long long>(mesh.triangles.size());

        std::cout << std::setprecision(17)
                  << "{\n"
                  << "  \"vertex_count\": " << mesh.vertices.size() << ",\n"
                  << "  \"face_count\": " << mesh.triangles.size() << ",\n"
                  << "  \"component_count\": " << component_count << ",\n"
                  << "  \"watertight\": " << (watertight ? "true" : "false") << ",\n"
                  << "  \"winding_consistent\": " << (winding_consistent ? "true" : "false") << ",\n"
                  << "  \"euler_number\": " << euler << ",\n"
                  << "  \"surface_area\": " << area << ",\n"
                  << "  \"volume\": ";
        if (watertight) std::cout << std::abs(signed_volume);
        else std::cout << "null";
        std::cout << "\n}\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "objTopology: " << exc.what() << '\n';
        return 1;
    }
}
