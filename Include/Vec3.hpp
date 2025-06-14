/* @file Vec3.hpp

    Definitions of vec3 structure.
    SPDX-License-Identifier: WTFPL

*/

#ifndef VEC3_HPP
#define VEC3_HPP

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_xorwow.h>
#include <rocrand_normal.h>

typedef struct vec3 {
    float x;
    float y;
    float z;
    constexpr __device__ vec3() : x(0), y(0), z(0) {}
    constexpr __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __device__ vec3 operator- () const {return vec3(-x,-y,-z);}

    __device__ vec3& operator+=(const vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __device__ vec3& operator*=(float t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    __device__ vec3& operator/=(float t) {
        return *this *= 1/t;
    }

    __device__ float length() const {
        return __fsqrt_rn(length_squared());
    }

    __device__ float length_squared() const {
        return  __fmaf_rn(x,x, __fmaf_rn(y,y, __fmul_rn(z,z)));
    }

    static __device__ vec3 random(rocrand_state_xorwow *rand) {
        return vec3(rocrand_normal(rand), rocrand_normal(rand), rocrand_normal(rand));
    }

    static __device__ vec3 random(float min, float max, rocrand_state_xorwow *rand) {
        float range = max - min;
        return vec3 (
                __fmaf_rn(rocrand_normal(rand),range, -min),
                __fmaf_rn(rocrand_normal(rand),range, -min),
                __fmaf_rn(rocrand_normal(rand),range, -min)
        );
    }
} vec3, color, point3;

__device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.x, t*v.y, t*v.z);
}

__device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__device__ inline vec3 operator/(const vec3& v, float t) {
    return (1/t) * v;
}

__device__ inline float dot(const vec3& u, const vec3& v) {
    return __fmaf_rn(u.x, v.x, __fmaf_rn(u.y, v.y, __fmul_rn(u.z, v.z)));
}

__device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(
            __fmaf_rn(u.y, v.z, -__fmul_rn(u.z, v.y)),
            __fmaf_rn(u.z, v.x, -__fmul_rn(u.x, v.z)),
            __fmaf_rn(u.x, v.y, -__fmul_rn(u.y, v.x))
    );
}

__device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ inline vec3 random_unit_vector(rocrand_state_xorwow *rand) {
    while (true) {
        auto p = vec3::random(-1,1, rand);
        auto lensq = p.length_squared();
        if (1e-8 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, rocrand_state_xorwow *rand) {
    vec3 on_unit_sphere = random_unit_vector(rand);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

#endif
