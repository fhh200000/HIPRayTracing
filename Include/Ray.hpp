/* @file Ray.hpp

    Definitions of ray structure.
    SPDX-License-Identifier: WTFPL

*/

#ifndef RAY_HPP
#define RAY_HPP

#include <Vec3.hpp>

typedef struct ray {
    vec3 direction;
    point3 origin;
    __device__ point3 at(float t) const {
        return point3 (
            __fmaf_rn(t, direction.x, origin.x),
            __fmaf_rn(t, direction.y, origin.y),
            __fmaf_rn(t, direction.z, origin.z)
        );
    }
    __device__ ray (vec3 ori, point3 dir) : origin(ori), direction (dir) {}
} ray;

#endif
