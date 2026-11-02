#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <string>
#include "objMesh.hpp"

using namespace meshio;

struct Sample { Point position, normal; };
double coord(Point a,int axis){return axis==0?a.x:(axis==1?a.y:a.z);}

struct Face { Point a,b,c,normal,centroid; std::size_t original; };
std::vector<Face> validFaces(const Mesh& mesh){
    std::vector<Face> faces;
    for(std::size_t i=0;i<mesh.triangles.size();++i){
        const auto& t=mesh.triangles[i];
        if(t.a>=mesh.vertices.size()||t.b>=mesh.vertices.size()||t.c>=mesh.vertices.size())
            throw std::invalid_argument("Invalid triangle index");
        Point a=mesh.vertices[t.a],b=mesh.vertices[t.b],c=mesh.vertices[t.c];
        Point n=cross(b-a,c-a); double len=norm(n);
        if(!(len>0)||!std::isfinite(len)) continue;
        faces.push_back({a,b,c,(1.0/len)*n,(1.0/3.0)*(a+b+c),i});
    }
    if(faces.empty()) throw std::invalid_argument("No valid nondegenerate triangles");
    return faces;
}
std::vector<Sample> sampleSurface(const Mesh& mesh,std::size_t N,std::uint64_t seed=42){
    auto faces=validFaces(mesh);
    std::vector<double> weights; weights.reserve(faces.size());
    for(const auto& f:faces) weights.push_back(norm(cross(f.b-f.a,f.c-f.a)));
    std::mt19937_64 rng(seed);
    std::discrete_distribution<std::size_t> choose(weights.begin(),weights.end());
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    std::vector<Sample> out;out.reserve(N);
    for(std::size_t i=0;i<N;++i){
        const auto& f=faces[choose(rng)];
        double s=std::sqrt(uniform(rng)),v=uniform(rng);
        out.push_back({(1-s)*f.a+(s*(1-v))*f.b+(s*v)*f.c,f.normal});
    }
    return out;
}

// Closest point on a nondegenerate triangle (Ericson's Voronoi-region algorithm).
Point closestPointTriangle(Point p,Point a,Point b,Point c){
    Point ab=b-a,ac=c-a,ap=p-a;
    double d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0&&d2<=0)return a;
    Point bp=p-b;
    double d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0&&d4<=d3)return b;
    double vc=d1*d4-d3*d2;
    if(vc<=0&&d1>=0&&d3<=0)return a+(d1/(d1-d3))*ab;
    Point cp=p-c;
    double d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0&&d5<=d6)return c;
    double vb=d5*d2-d1*d6;
    if(vb<=0&&d2>=0&&d6<=0)return a+(d2/(d2-d6))*ac;
    double va=d3*d6-d5*d4;
    if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)
        return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b);
    double denom=1.0/(va+vb+vc);
    return a+(vb*denom)*ab+(vc*denom)*ac;
}

struct AABB {
    Point lo{INFINITY,INFINITY,INFINITY},hi{-INFINITY,-INFINITY,-INFINITY};
    void add(Point p){lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
        hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);}
    double distance2(Point p)const{
        auto delta=[](double x,double a,double b){return x<a?a-x:(x>b?x-b:0.0);};
        double x=delta(p.x,lo.x,hi.x),y=delta(p.y,lo.y,hi.y),z=delta(p.z,lo.z,hi.z);
        return x*x+y*y+z*z;
    }
};
struct Nearest {Point position{},normal{};double distance2=INFINITY;std::size_t triangle=0;};
class TriangleBVH {
    struct Node {AABB box;std::size_t begin=0,end=0;int left=-1,right=-1;};
    std::vector<Face> faces_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
    int build(std::size_t begin,std::size_t end){
        int id=static_cast<int>(nodes_.size());nodes_.emplace_back();
        AABB box,centers;
        for(std::size_t i=begin;i<end;++i){const auto& f=faces_[order_[i]];
            box.add(f.a);box.add(f.b);box.add(f.c);centers.add(f.centroid);}
        nodes_[id].box=box;nodes_[id].begin=begin;nodes_[id].end=end;
        if(end-begin<=8)return id;
        Point span=centers.hi-centers.lo;
        int axis=span.x>=span.y&&span.x>=span.z?0:(span.y>=span.z?1:2);
        std::size_t mid=begin+(end-begin)/2;
        std::nth_element(order_.begin()+begin,order_.begin()+mid,order_.begin()+end,
            [&](std::size_t i,std::size_t j){return coord(faces_[i].centroid,axis)<coord(faces_[j].centroid,axis);});
        int left=build(begin,mid),right=build(mid,end);
        nodes_[id].left=left;nodes_[id].right=right;
        return id;
    }
    void visit(int id,Point p,Nearest& best)const{
        const Node& n=nodes_[id];
        if(n.box.distance2(p)>best.distance2)return;
        if(n.left<0){
            for(std::size_t i=n.begin;i<n.end;++i){const Face& f=faces_[order_[i]];
                Point q=closestPointTriangle(p,f.a,f.b,f.c);
                double d2=norm2(q-p);
                if(d2<best.distance2){best={q,f.normal,d2,f.original};}
            }
            return;
        }
        double dl=nodes_[n.left].box.distance2(p),dr=nodes_[n.right].box.distance2(p);
        if(dl<dr){visit(n.left,p,best);visit(n.right,p,best);}
        else {visit(n.right,p,best);visit(n.left,p,best);}
    }
public:
    explicit TriangleBVH(const Mesh& mesh):faces_(validFaces(mesh)){
        order_.resize(faces_.size());std::iota(order_.begin(),order_.end(),0);
        nodes_.reserve(2*faces_.size());build(0,faces_.size());
    }
    Nearest nearest(Point p)const{Nearest best;visit(0,p,best);return best;}
};


struct PointErrors {
    double distance;     // Euclidean positional error
    double angleDegrees; // normal error at the corresponding closest point
};

std::vector<PointErrors> computeErrors(const std::vector<Sample>& samples,
                                       const TriangleBVH& target,
                                       bool unorientedNormals){
    std::vector<PointErrors> errors;
    errors.reserve(samples.size());
    const double radToDeg=180.0/std::acos(-1.0);
    for(const auto& s:samples){
        Nearest q=target.nearest(s.position);
        double cosine=dot(s.normal,q.normal);
        if(unorientedNormals) cosine=std::abs(cosine);
        cosine=std::clamp(cosine,-1.0,1.0);
        errors.push_back({std::sqrt(q.distance2),std::acos(cosine)*radToDeg});
    }
    return errors;
}

struct DirectedMetrics {
    double L1=0.0;
    double L2=0.0;
    double Linf=0.0;
    double meanAngleDegrees=0.0;
};

DirectedMetrics metricsFromErrors(const std::vector<PointErrors>& errors){
    if(errors.empty()) throw std::invalid_argument("Empty error set");
    double sum1=0.0,sum2=0.0,sumAngle=0.0,maxD=0.0;
    for(const auto& e:errors){
        sum1+=e.distance;
        sum2+=e.distance*e.distance;
        maxD=std::max(maxD,e.distance);
        sumAngle+=e.angleDegrees;
    }
    const double n=static_cast<double>(errors.size());
    return {sum1/n,std::sqrt(sum2/n),maxD,sumAngle/n};
}

// Keep the fraction 'keepFraction' having the smallest value according to
// the selected metric. The trimming is performed independently for each metric.
template <typename Getter>
double trimmedLp(const std::vector<PointErrors>& errors,double keepFraction,
                 double p,Getter getter){
    if(errors.empty()) throw std::invalid_argument("Empty error set");
    std::size_t k=static_cast<std::size_t>(std::ceil(keepFraction*errors.size()));
    k=std::clamp<std::size_t>(k,1,errors.size());
    std::vector<double> v; v.reserve(errors.size());
    for(const auto& e:errors) v.push_back(getter(e));
    if(k<v.size()) std::nth_element(v.begin(),v.begin()+k,v.end());
    if(std::isinf(p)) return *std::max_element(v.begin(),v.begin()+k);
    double sum=0.0;
    for(std::size_t i=0;i<k;++i) sum+=std::pow(v[i],p);
    return std::pow(sum/static_cast<double>(k),1.0/p);
}

DirectedMetrics trimmedMetrics(const std::vector<PointErrors>& errors,double keepFraction){
    auto distance=[](const PointErrors& e){return e.distance;};
    auto angle=[](const PointErrors& e){return e.angleDegrees;};
    return {trimmedLp(errors,keepFraction,1.0,distance),
            trimmedLp(errors,keepFraction,2.0,distance),
            trimmedLp(errors,keepFraction,std::numeric_limits<double>::infinity(),distance),
            trimmedLp(errors,keepFraction,1.0,angle)};
}

struct ComparisonMetrics {
    DirectedMetrics from1to2;
    DirectedMetrics from2to1;
    DirectedMetrics trimmed1to2; // same, restricted to the best keepFraction
    DirectedMetrics trimmed2to1;

    double L1Sym=0.0;   // max of the two directed L1 distances
    double L2Sym=0.0;   // max of the two directed L2 distances
    double LinfSym=0.0; // sampled-source Hausdorff estimate
    double angleSymDegrees=0.0; // mean of the two directed mean angles

    double keepFraction=0.9;
    double trimmedL1Sym=0.0;
    double trimmedL2Sym=0.0;
    double trimmedLinfSym=0.0;
    double trimmedAngleSymDegrees=0.0;
};

ComparisonMetrics compareSurfaces(const Mesh& mesh1,const Mesh& mesh2,std::size_t N,
                                  double keepFraction=0.90,
                                  bool unorientedNormals=false,
                                  std::uint64_t seed=42){
    if(N==0) throw std::invalid_argument("N must be > 0");
    if(!(keepFraction>0.0 && keepFraction<=1.0))
        throw std::invalid_argument("keepFraction must belong to (0,1]");

    auto s1=sampleSurface(mesh1,N,seed);
    auto s2=sampleSurface(mesh2,N,seed+1);
    TriangleBVH bvh1(mesh1),bvh2(mesh2);

    auto e12=computeErrors(s1,bvh2,unorientedNormals);
    auto e21=computeErrors(s2,bvh1,unorientedNormals);
    auto m12=metricsFromErrors(e12);
    auto m21=metricsFromErrors(e21);

    ComparisonMetrics r;
    r.from1to2=m12;
    r.from2to1=m21;
    r.L1Sym=std::max(m12.L1,m21.L1);
    r.L2Sym=std::max(m12.L2,m21.L2);
    r.LinfSym=std::max(m12.Linf,m21.Linf);
    r.angleSymDegrees=0.5*(m12.meanAngleDegrees+m21.meanAngleDegrees);
    r.keepFraction=keepFraction;

    // "Best keepFraction" is selected separately in each direction and metric.
    auto t12=trimmedMetrics(e12,keepFraction);
    auto t21=trimmedMetrics(e21,keepFraction);
    r.trimmed1to2=t12;
    r.trimmed2to1=t21;

    r.trimmedL1Sym=std::max(t12.L1,t21.L1);
    r.trimmedL2Sym=std::max(t12.L2,t21.L2);
    r.trimmedLinfSym=std::max(t12.Linf,t21.Linf);
    r.trimmedAngleSymDegrees=0.5*(t12.meanAngleDegrees+t21.meanAngleDegrees);
    return r;
}

// Machine-readable output: directed (1->2, 2->1) raw and trimmed metrics.
void printJson(const ComparisonMetrics& r,std::ostream& out=std::cout){
    auto dir=[&](const char* name,const DirectedMetrics& m,bool last){
        out<<"  \""<<name<<"\": {\"L1\": "<<m.L1<<", \"L2\": "<<m.L2
           <<", \"Linf\": "<<m.Linf<<", \"angle_deg\": "<<m.meanAngleDegrees<<"}"<<(last?"\n":",\n");
    };
    const auto chamferL2=[](const DirectedMetrics& a,const DirectedMetrics& b){
        return 0.5*(a.L2*a.L2+b.L2*b.L2);
    };
    out<<std::setprecision(17)<<"{\n";
    out<<"  \"keep_fraction\": "<<r.keepFraction<<",\n";
    dir("from1to2",r.from1to2,false);
    dir("from2to1",r.from2to1,false);
    dir("trimmed1to2",r.trimmed1to2,false);
    dir("trimmed2to1",r.trimmed2to1,false);
    const double trimL2=chamferL2(r.trimmed1to2,r.trimmed2to1);
    const double fullL2=chamferL2(r.from1to2,r.from2to1);
    out<<"  \"chamfer_l1_trim90\": "<<0.5*(r.trimmed1to2.L1+r.trimmed2to1.L1)<<",\n"
       <<"  \"chamfer_l2_trim90\": "<<trimL2<<",\n"
       <<"  \"chamfer_rmse_trim90\": "<<std::sqrt(trimL2)<<",\n"
       <<"  \"hausdorff_trim90\": "<<std::max(r.trimmed1to2.Linf,r.trimmed2to1.Linf)<<",\n"
       <<"  \"mean_distance_pred_to_gt_trim90\": "<<r.trimmed1to2.L1<<",\n"
       <<"  \"mean_distance_gt_to_pred_trim90\": "<<r.trimmed2to1.L1<<",\n"
       <<"  \"max_distance_pred_to_gt_trim90\": "<<r.trimmed1to2.Linf<<",\n"
       <<"  \"max_distance_gt_to_pred_trim90\": "<<r.trimmed2to1.Linf<<",\n"
       <<"  \"normal_angle_pred_to_gt_trim90_deg\": "<<r.trimmed1to2.meanAngleDegrees<<",\n"
       <<"  \"normal_angle_gt_to_pred_trim90_deg\": "<<r.trimmed2to1.meanAngleDegrees<<",\n"
       <<"  \"normal_angle_mean_trim90_deg\": "
       <<0.5*(r.trimmed1to2.meanAngleDegrees+r.trimmed2to1.meanAngleDegrees)<<",\n"
       <<"  \"chamfer_l1\": "<<0.5*(r.from1to2.L1+r.from2to1.L1)<<",\n"
       <<"  \"chamfer_l2\": "<<fullL2<<",\n"
       <<"  \"chamfer_rmse\": "<<std::sqrt(fullL2)<<",\n"
       <<"  \"hausdorff\": "<<std::max(r.from1to2.Linf,r.from2to1.Linf)<<",\n"
       <<"  \"mean_distance_pred_to_gt\": "<<r.from1to2.L1<<",\n"
       <<"  \"mean_distance_gt_to_pred\": "<<r.from2to1.L1<<",\n"
       <<"  \"max_distance_pred_to_gt\": "<<r.from1to2.Linf<<",\n"
       <<"  \"max_distance_gt_to_pred\": "<<r.from2to1.Linf<<",\n"
       <<"  \"normal_angle_pred_to_gt_deg\": "<<r.from1to2.meanAngleDegrees<<",\n"
       <<"  \"normal_angle_gt_to_pred_deg\": "<<r.from2to1.meanAngleDegrees<<",\n"
       <<"  \"normal_angle_mean_deg\": "
       <<0.5*(r.from1to2.meanAngleDegrees+r.from2to1.meanAngleDegrees)<<"\n"
       <<"}\n";
}

void printMetrics(const ComparisonMetrics& r,std::ostream& out=std::cout){
    out<<std::setprecision(10);
    out<<"L1  1 -> 2       : "<<r.from1to2.L1<<'\n';
    out<<"L1  2 -> 1       : "<<r.from2to1.L1<<'\n';
    out<<"L1  symmetric    : "<<r.L1Sym<<"  (max)\n\n";

    out<<"L2  1 -> 2       : "<<r.from1to2.L2<<'\n';
    out<<"L2  2 -> 1       : "<<r.from2to1.L2<<'\n';
    out<<"L2  symmetric    : "<<r.L2Sym<<"  (max)\n\n";

    out<<"Linf 1 -> 2      : "<<r.from1to2.Linf<<'\n';
    out<<"Linf 2 -> 1      : "<<r.from2to1.Linf<<'\n';
    out<<"Linf symmetric   : "<<r.LinfSym<<"  (Hausdorff estimate)\n\n";

    out<<"Mean angle sym.  : "<<r.angleSymDegrees<<" deg\n\n";

    out<<"Metrics on best "<<(100.0*r.keepFraction)<<"%:\n";
    out<<"L1   symmetric   : "<<r.trimmedL1Sym<<'\n';
    out<<"L2   symmetric   : "<<r.trimmedL2Sym<<'\n';
    out<<"Linf symmetric   : "<<r.trimmedLinfSym<<'\n';
    out<<"Mean angle sym.  : "<<r.trimmedAngleSymDegrees<<" deg\n";
}


// struct DirectedStats {double position=0,angleDegrees=0,maxDistance=0;};
// struct DistanceResult {
//     DirectedStats from1to2,from2to1;
//     double position=0,angleDegrees=0,sampledHausdorff=0;
// };
// DirectedStats directed(const std::vector<Sample>& samples,const TriangleBVH& target,
//                        double p,bool unoriented){
//     double sum=0,angleSum=0,maxDistance=0;
//     for(const auto& s:samples){
//         Nearest q=target.nearest(s.position);
//         double d=std::sqrt(q.distance2);
//         maxDistance=std::max(maxDistance,d);
//         if(std::isinf(p))sum=std::max(sum,d);else sum+=std::pow(d,p);
//         double cosine=dot(s.normal,q.normal);
//         if(unoriented)cosine=std::abs(cosine);
//         angleSum+=std::acos(std::clamp(cosine,-1.0,1.0));
//     }
//     return {std::isinf(p)?sum:std::pow(sum/samples.size(),1.0/p),
//         angleSum/samples.size()*180.0/std::acos(-1.0),maxDistance};
// }
// DistanceResult compareSurfaces(const Mesh& a,const Mesh& b,std::size_t N,
//                                double p=2.0,bool unorientedNormals=true,std::uint64_t seed=42){
//     if(N==0||!(p>=1.0)||std::isnan(p))throw std::invalid_argument("Require N>0 and p>=1");
//     auto s1=sampleSurface(a,N,seed),s2=sampleSurface(b,N,seed+1);
//     TriangleBVH bvh1(a),bvh2(b);
//     auto ab=directed(s1,bvh2,p,unorientedNormals);
//     auto ba=directed(s2,bvh1,p,unorientedNormals);
//     double pos=std::isinf(p)?std::max(ab.position,ba.position):
//         std::pow((std::pow(ab.position,p)+std::pow(ba.position,p))/2.0,1.0/p);
//     return {ab,ba,pos,(ab.angleDegrees+ba.angleDegrees)/2.0,
//         std::max(ab.maxDistance,ba.maxDistance)};
// }

///////////////////////////////////////////////////////////////////////////////
// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char* argv[] )
{
  try {
    std::string mesh1_name, mesh2_name;
    std::size_t sample_count = 10000;
    bool json = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--json") {
        json = true;
      } else if (arg == "--i1" || arg == "-1" || arg == "--i2" || arg == "-2" ||
                 arg == "-n" || arg == "--nb-samples") {
        if (++i >= argc) throw std::invalid_argument("Missing value after " + arg);
        const std::string value = argv[i];
        if (arg == "--i1" || arg == "-1") mesh1_name = value;
        else if (arg == "--i2" || arg == "-2") mesh2_name = value;
        else sample_count = static_cast<std::size_t>(std::stoull(value));
      } else if (!arg.empty() && arg[0] != '-') {
        positional.push_back(arg);
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    }
    if (mesh1_name.empty() && !positional.empty()) mesh1_name = positional[0];
    if (mesh2_name.empty() && positional.size() > 1) mesh2_name = positional[1];
    if (mesh1_name.empty() || mesh2_name.empty() || sample_count == 0) {
      std::cerr << "Usage: objCompare <mesh1.obj> <mesh2.obj> [-n samples] [--json]\n";
      std::cerr << "   or: objCompare --i1 mesh1.obj --i2 mesh2.obj [-n samples] [--json]\n";
      return 2;
    }

    const Mesh a = readObjMesh(mesh1_name);
    const Mesh b = readObjMesh(mesh2_name);
    const double keepFraction = 0.90;
    const bool unorientedNormals = true;
    const auto result = compareSurfaces(a, b, sample_count, keepFraction, unorientedNormals);
    if (json) printJson(result); else printMetrics(result);
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "objCompare: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
