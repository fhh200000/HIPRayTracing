/* @file Kernel.cpp

    Definitions related to HIP device.
    SPDX-License-Identifier: WTFPL

*/

#include <Kernel.hpp>
#include <Window.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>
#include <Hittable.hpp>
#include <Sphere.hpp>
#include <rocrand/rocrand_xorwow.h>
#include <rocrand_normal.h>

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

__device__ constexpr float maximum_color_depth = WINDOW_COLOR_DEPTH-0.01f;

__device__ constexpr sphere world[] = {
    sphere(point3(0,0,-1), 0.5),
    sphere(point3(0,-100.5,-1), 100),
};

__device__ bool world_hit(const ray& r, hit_record& rec)
{
    float ray_tmin = 0;
    float ray_tmax = INFINITY;
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_tmax;

    for (const auto& object : world) {
        if (object.hit(r, ray_tmin, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ color ray_color(const ray& r) {
    hit_record rec;
    if (world_hit(r, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    vec3 unit_direction = unit_vector(r.direction);
    float a = 0.5f*(unit_direction.y + 1.0f);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__device__ __inline__ vec3 sample_square(rocrand_state_xorwow *state) {
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return vec3(rocrand_normal(state)-0.5, rocrand_normal(state)-0.5, 0);
}

__global__ void RayTracingKernel(pixel_t *output_buffer)
{
    unsigned int index = __fmaf_rn(__fmaf_rn(blockIdx.y, blockDim.y, threadIdx.y), WINDOW_WIDTH, __fmaf_rn(blockIdx.x, blockDim.x, threadIdx.x));

    rocrand_state_xorwow rand_state;
    rocrand_init(index, 0, 0, &rand_state);

    point3 pixel_center = pixel00_loc + (__fmaf_rn(blockIdx.x, blockDim.x, threadIdx.x) * pixel_delta_u) + (__fmaf_rn(blockIdx.y, blockDim.y, threadIdx.y) * pixel_delta_v);
    color pixel_color;

    for (size_t it=0; it<SAMPLES_PER_PIXEL;++it) {
        vec3 random_square = sample_square(&rand_state);
        vec3 sampled_center = pixel_center + random_square.x*pixel_delta_u + random_square.y*pixel_delta_v;
        vec3 ray_direction = sampled_center - camera_center;
        ray r(camera_center, ray_direction);
        pixel_color += ray_color(r);
    }
    output_buffer[index].R = maximum_color_depth/SAMPLES_PER_PIXEL*pixel_color.x;
    output_buffer[index].G = maximum_color_depth/SAMPLES_PER_PIXEL*pixel_color.y;
    output_buffer[index].B = maximum_color_depth/SAMPLES_PER_PIXEL*pixel_color.z;

}
