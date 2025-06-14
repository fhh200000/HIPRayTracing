/* @file Hittable.hpp

    Abstract definitions of hittable shapes.
    SPDX-License-Identifier: WTFPL

*/

#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include <hip/hip_runtime.h>
#include <Vec3.hpp>
#include <Ray.hpp>

struct hit_record {
    point3 p;
    vec3 normal;
    float t;
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct hittable {
    __device__ virtual bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const = 0;
};

#endif
