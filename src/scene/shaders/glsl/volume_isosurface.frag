#version 450

#include "color.glsl"

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 proj;
    float time;
    uint flags;
} mvp;

layout(set = 1, binding = 0) uniform texture3D tex;
layout(set = 1, binding = 1) uniform sampler samp;
/* Unused by the isosurface shader: real hardware depth (below) replaces the manual
 * depth-texture occlusion test used by the translucent composite/MIP/slice render modes.
 * Declared to keep the set = 1 descriptor layout identical across all volume shader variants. */
layout(set = 1, binding = 3) uniform texture2D depthTex;
layout(set = 1, binding = 4) uniform texture2D transferTex;

layout(set = 1, binding = 2) uniform VolumeParams {
    vec4 clip_min;
    vec4 clip_max;
    vec4 clip_plane;
    vec4 clip_plane_params;
    vec4 params;
    vec4 slice;
    vec4 bounds_min;
    vec4 bounds_max;
    vec4 axis_order;
    vec4 axis_flip;
    vec4 value_range;
    vec4 occlusion;
    vec4 texture_params;
    vec4 iso_params;   /* threshold, mode (0=below,1=above), gradient step, use_value_color */
    vec4 iso_color;    /* flat base color rgba */
    vec4 iso_light;    /* light_dir.xyz, shininess */
    vec4 iso_material; /* ambient, diffuse, specular, reserved */
} volume;

layout(location = 0) in vec3 fragUVW;
layout(location = 1) in vec3 fragObj;
layout(location = 0) out vec4 outColor;

const int MAX_STEPS = 1024;
const int BISECTION_STEPS = 6;

float safe_inv(float v)
{
    if (abs(v) < 1e-6) {
        return v < 0.0 ? -1e6 : 1e6;
    }
    return 1.0 / v;
}

bool ray_box(vec3 ro, vec3 rd, vec3 box_min, vec3 box_max, out float t0, out float t1)
{
    vec3 inv_rd = vec3(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
    vec3 t_near = (box_min - ro) * inv_rd;
    vec3 t_far = (box_max - ro) * inv_rd;
    vec3 t_min = min(t_near, t_far);
    vec3 t_max = max(t_near, t_far);
    t0 = max(max(t_min.x, t_min.y), t_min.z);
    t1 = min(min(t_max.x, t_max.y), t_max.z);
    return t1 >= max(t0, 0.0);
}

vec3 camera_object()
{
    mat4 inv_model = inverse(mvp.model);
    mat4 inv_view = inverse(mvp.view);
    return (inv_model * inv_view * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
}

vec3 object_to_uvw(vec3 pos)
{
    vec3 extent = max(volume.bounds_max.xyz - volume.bounds_min.xyz, vec3(1e-6));
    return (pos - volume.bounds_min.xyz) / extent;
}

vec3 object_dir_to_uvw(vec3 dir)
{
    vec3 extent = max(volume.bounds_max.xyz - volume.bounds_min.xyz, vec3(1e-6));
    return dir / extent;
}

vec3 uvw_to_object(vec3 uvw)
{
    return mix(volume.bounds_min.xyz, volume.bounds_max.xyz, uvw);
}

float axis_value(int axis, vec3 value)
{
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

vec3 texture_uvw(vec3 uvw)
{
    int ax0 = int(clamp(volume.axis_order.x, 0.0, 2.0));
    int ax1 = int(clamp(volume.axis_order.y, 0.0, 2.0));
    int ax2 = int(clamp(volume.axis_order.z, 0.0, 2.0));
    vec3 out_uvw = vec3(axis_value(ax0, uvw), axis_value(ax1, uvw), axis_value(ax2, uvw));
    out_uvw = mix(out_uvw, vec3(1.0) - out_uvw, step(vec3(0.5), volume.axis_flip.xyz));
    return clamp(out_uvw, vec3(0.0), vec3(1.0));
}

float sample_value(vec3 uvw)
{
    return texture(sampler3D(tex, samp), texture_uvw(uvw)).r;
}

vec4 transfer_value(float value)
{
    float denom = max(volume.value_range.y - volume.value_range.x, 1e-12);
    float t = clamp((value - volume.value_range.x) / denom, 0.0, 1.0);
    return semanticColorToLinear(texture(sampler2D(transferTex, samp), vec2(t, 0.5)));
}

bool inside_clip_plane(vec3 uvw)
{
    if (volume.clip_plane_params.x < 0.5) {
        return true;
    }
    float side = dot(volume.clip_plane.xyz, uvw) + volume.clip_plane.w;
    return volume.clip_plane_params.y > 0.5 ? side >= -1e-6 : side <= 1e-6;
}

float projected_depth(vec3 uvw)
{
    vec3 pos = uvw_to_object(uvw);
    vec4 clip = mvp.proj * mvp.view * mvp.model * vec4(pos, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    return clamp(0.5 * (clip.z / clip.w) + 0.5, 0.0, 1.0);
}

/* Central-difference gradient of the scalar field at a UVW position, in UVW space. */
vec3 estimate_gradient(vec3 uvw, float h)
{
    float dx = sample_value(uvw + vec3(h, 0.0, 0.0)) - sample_value(uvw - vec3(h, 0.0, 0.0));
    float dy = sample_value(uvw + vec3(0.0, h, 0.0)) - sample_value(uvw - vec3(0.0, h, 0.0));
    float dz = sample_value(uvw + vec3(0.0, 0.0, h)) - sample_value(uvw - vec3(0.0, 0.0, h));
    return vec3(dx, dy, dz);
}

/* Whether `value` is on the "inside" side of the threshold for the configured crossing mode. */
bool is_inside(float value, float threshold, float mode)
{
    return mode < 0.5 ? value < threshold : value > threshold;
}

void main()
{
    vec3 ro_obj = camera_object();
    vec3 rd_obj = normalize(fragObj - ro_obj);
    vec3 ro = object_to_uvw(ro_obj);
    vec3 rd = object_dir_to_uvw(rd_obj);

    float proxy_t0 = 0.0;
    float proxy_t1 = 0.0;
    if (!ray_box(ro, rd, vec3(0.0), vec3(1.0), proxy_t0, proxy_t1)) {
        discard;
    }

    vec3 box_min = volume.clip_min.xyz;
    vec3 box_max = volume.clip_max.xyz;
    float t0 = 0.0;
    float t1 = 0.0;
    if (!ray_box(ro, rd, box_min, box_max, t0, t1)) {
        discard;
    }

    int steps = int(clamp(volume.params.z, 1.0, float(MAX_STEPS)));
    float start_t = max(t0, 0.0);
    float end_t = t1;
    if (end_t <= start_t) {
        discard;
    }

    float threshold = float(volume.iso_params.x);
    float mode = volume.iso_params.y;

    float ray_length = end_t - start_t;
    float step_len = ray_length / float(steps);

    /* The baseline just before the ray enters the sampled field is treated as "outside" the
     * threshold region for both crossing directions. Without this, a ray whose very first
     * sample is already past the threshold (e.g. the camera is close to, or inside, a region
     * that satisfies the threshold once it's raised high enough that most of the field
     * qualifies) would never register an in-ray transition and the surface would incorrectly
     * disappear instead of rendering right at the entry point. */
    bool have_prev = true;
    float prev_t = start_t;
    vec3 prev_uvw = ro + rd * start_t;
    float prev_value = 0.0;
    bool prev_inside = false;

    bool hit = false;
    float hit_t = 0.0;

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= steps) {
            break;
        }
        float t = start_t + step_len * (float(i) + 0.5);
        vec3 uvw = ro + rd * t;

        if (!inside_clip_plane(uvw)) {
            /* Resuming after a clip-plane gap is treated the same as a fresh ray entry. */
            have_prev = true;
            prev_t = t;
            prev_inside = false;
            continue;
        }

        float value = sample_value(uvw);
        bool inside = is_inside(value, threshold, mode);

        if (have_prev && inside != prev_inside) {
            /* Threshold crossing between prev_t and t: refine with bisection. */
            float lo_t = prev_t;
            float hi_t = t;
            bool lo_inside = prev_inside;
            for (int b = 0; b < BISECTION_STEPS; b++) {
                float mid_t = 0.5 * (lo_t + hi_t);
                vec3 mid_uvw = ro + rd * mid_t;
                float mid_value = sample_value(mid_uvw);
                bool mid_inside = is_inside(mid_value, threshold, mode);
                if (mid_inside == lo_inside) {
                    lo_t = mid_t;
                } else {
                    hi_t = mid_t;
                }
            }
            hit_t = 0.5 * (lo_t + hi_t);
            hit = true;
            break;
        }

        have_prev = true;
        prev_t = t;
        prev_uvw = uvw;
        prev_value = value;
        prev_inside = inside;
    }

    if (!hit) {
        discard;
    }

    vec3 hit_uvw = ro + rd * hit_t;

    float h = volume.iso_params.z;
    if (h <= 0.0) {
        /* Automatic gradient step: about one texel of a 128^3 field; coarse but robust
         * without knowing the exact field resolution in the shader. */
        h = 1.0 / 128.0;
    }
    vec3 gradient = estimate_gradient(hit_uvw, h);
    vec3 normal = length(gradient) > 1e-8 ? normalize(gradient) : vec3(0.0, 0.0, 1.0);
    /* Orient the normal to face the camera, robust to the sign convention of the crossing. */
    if (dot(normal, rd) > 0.0) {
        normal = -normal;
    }

    vec4 base = volume.iso_params.w > 0.5 ? transfer_value(sample_value(hit_uvw))
                                           : semanticColorToLinear(volume.iso_color);

    vec3 view_dir = normalize(-rd);
    vec3 light_dir = normalize(volume.iso_light.xyz);
    float shininess = max(volume.iso_light.w, 1.0);
    float ambient = volume.iso_material.x;
    float diffuse_coef = volume.iso_material.y;
    float specular_coef = volume.iso_material.z;

    float diffuse = max(dot(normal, light_dir), 0.0);
    vec3 half_dir = normalize(light_dir + view_dir);
    float specular = pow(max(dot(normal, half_dir), 0.0), shininess);

    vec3 shaded = base.rgb * (ambient + diffuse_coef * diffuse) + specular_coef * specular;
    outColor = vec4(shaded, base.a);

    if (outColor.a <= 0.0) {
        discard;
    }

    gl_FragDepth = projected_depth(hit_uvw);
}
