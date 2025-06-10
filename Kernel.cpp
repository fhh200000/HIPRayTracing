/* @file Kernel.cpp

    Definitions related to HIP device.
    SPDX-License-Identifier: WTFPL

*/

#include <Kernel.hpp>
#include <Window.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>

// Camera
__device__ constexpr float focal_length = 1.0;
__device__ constexpr float viewport_height = 2.0;
__device__ constexpr float viewport_width = viewport_height * (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;
__device__ constexpr point3 camera_center = point3(0,0,0);

// Calculate the vectors across the horizontal and down the vertical viewport edges.
__device__ constexpr point3 viewport_u = vec3(viewport_width, 0, 0);
__device__ constexpr point3 viewport_v = vec3(0, -viewport_height, 0);

// Calculate the horizontal and vertical delta vectors from pixel to pixel.
__device__ constexpr point3 pixel_delta_u = vec3(viewport_u.x/WINDOW_WIDTH, viewport_u.y/WINDOW_WIDTH, viewport_u.z/WINDOW_WIDTH);
__device__ constexpr point3 pixel_delta_v = vec3(viewport_v.x/WINDOW_HEIGHT, viewport_v.y/WINDOW_HEIGHT, viewport_v.z/WINDOW_HEIGHT);

// Calculate the location of the upper left pixel.
__device__ constexpr point3 viewport_upper_left = vec3 (
    camera_center.x - viewport_u.x/2 - viewport_v.x/2,
    camera_center.y - viewport_u.y/2 - viewport_v.y/2,
    camera_center.z - focal_length - viewport_u.z/2 - viewport_v.z/2
);

__device__ constexpr point3 pixel00_loc = vec3 (
    viewport_upper_left.x + 0.5 * (pixel_delta_u.x + pixel_delta_v.x),
    viewport_upper_left.y + 0.5 * (pixel_delta_u.y + pixel_delta_v.y),
    viewport_upper_left.z + 0.5 * (pixel_delta_u.z + pixel_delta_v.z)
);


__device__ float hit_sphere(const point3& center, float radius, const ray& r) {
    vec3 oc = center - r.origin;
    float a = dot(r.direction, r.direction);
    float b = -2.0 * dot(r.direction, oc);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;

    if (discriminant < 0) {
        return -1.0f;
    } else {
        return (-b - __fsqrt_rn(discriminant) ) / (2.0*a);
    }
}

__device__ color ray_color(const ray& r) {
    float t = hit_sphere(point3(0,0,-1), 0.5, r);
    if (t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
        return 0.5*color(N.x+1, N.y+1, N.z+1);
    }
    vec3 unit_direction = unit_vector(r.direction);
    float a = 0.5f*(unit_direction.y + 1.0f);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void RayTracingKernel(pixel_t *output_buffer)
{
    unsigned int index = __fmaf_rn(__fmaf_rn(blockIdx.y, blockDim.y, threadIdx.y), WINDOW_WIDTH, __fmaf_rn(blockIdx.x, blockDim.x, threadIdx.x));

    point3 pixel_center = pixel00_loc + (__fmaf_rn(blockIdx.x, blockDim.x, threadIdx.x) * pixel_delta_u) + (__fmaf_rn(blockIdx.y, blockDim.y, threadIdx.y) * pixel_delta_v);
    vec3 ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);

    color pixel_color = ray_color(r);
    output_buffer[index].R = (uint8_t)(255.99f*pixel_color.x);
    output_buffer[index].G = (uint8_t)(255.99f*pixel_color.y);
    output_buffer[index].B = (uint8_t)(255.99f*pixel_color.z);

}


