/* @file Sphere.hpp

    Definition of sphere.
    SPDX-License-Identifier: WTFPL

*/

#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <Hittable.hpp>
struct sphere : public hittable {
public:
    point3 center;
    float radius;
    constexpr __device__ sphere() : center (vec3()), radius (0.f) {}
    constexpr __device__ sphere(const point3& center, float radius) : center(center), radius(radius>0.f?radius:0.f){}
    __device__ bool hit(const ray & r, float ray_tmin, float ray_tmax, hit_record & rec) const override {
        vec3 oc = center - r.origin;
        float a = r.direction.length_squared();
        float h = dot(r.direction, oc);
        float c = oc.length_squared() - radius*radius;
        float discriminant = h*h - a*c;

        if (discriminant < 0) {
            return false;
        }
        float sqrtd = __fsqrt_rn(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }
};

#endif
