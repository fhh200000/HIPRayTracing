/* @file Vec3.hpp

    Definitions of vec3 structure.
    SPDX-License-Identifier: WTFPL

*/

#ifndef VEC3_HPP
#define VEC3_HPP

#include <hip/hip_runtime.h>

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

#endif
