// Tobias Czyzewski Total War Shader
// =================================

// #version 330
//
// // Vertex shader output
// struct V2F {
//   vec3 normal;               // interpolated normal
//   vec3 tangent;              // interpolated tangent
//   vec3 bitangent;            // interpolated bitangent
//   vec3 position;             // interpolated position
//   vec4 color[1];             // interpolated vertex colors (color0)
//   vec2 tex_coord;            // interpolated texture coordinates (uv0)
//   vec2 multi_tex_coord[8];   // interpolated texture coordinates (uv0-uv7)
// };

//- Enable alpha blending
//: state blend over

/////////////////////////
// Parameters
/////////////////////////

//: param auto camera_view_matrix_it
uniform mat4 camera_view_matrix_it;

mat4 vMatrixI = transpose(camera_view_matrix_it);

//: param auto channel_diffuse
uniform sampler2D s_diffuse_color;
//: param auto channel_glossiness
uniform sampler2D s_smoothness;
//: param auto texture_normal
uniform sampler2D s_normal_tex;
//: param auto channel_normal
uniform sampler2D s_normal_map;
//: param auto channel_height
uniform sampler2D s_height_map;
//: param auto channel_specularlevel
uniform sampler2D s_reflectivity;
//: param auto channel_specular
uniform sampler2D s_specular_color;
//: param auto channel_opacity
uniform sampler2D s_opacity;
//: param auto texture_ambientocclusion
uniform sampler2D s_ambient_occlusion_tex;
//: param auto channel_ambientocclusion
uniform sampler2D s_ambient_occlusion;
//: param auto channel_user0
uniform sampler2D s_mask1;
//: param auto channel_user1
uniform sampler2D s_mask2;
//: param auto channel_user2
uniform sampler2D s_mask3;
//: param auto channel_user3
uniform sampler2D s_decal_diffuse;
//: param auto channel_user4
uniform sampler2D s_decal_normal;
//: param auto channel_user5
uniform sampler2D s_decal_mask;
//: param auto channel_user6
uniform sampler2D s_decal_dirtmask;
//: param auto channel_user7
uniform sampler2D s_alpha_mask;

//: param custom {
//:   "default": 0,
//:   "label": "Shader Version",
//:   "widget": "combobox",
//:   "values": {
//:     "v1.4": 0
//:   }
//: }
uniform int i_version;
//: param custom {
//:   "default": 8,
//:   "label": "Technique",
//:   "widget": "combobox",
//:   "values": {
//:     "Ambient Occlusion": 0,
//:     "Vertex Color": 1,
//:     "Custom Terrain": 2,
//:     "Decal Dirt": 3,
//:     "Diffuse": 4,
//:     "Full Ambient Occlusion": 5,
//:     "Full Dirtmap": 6,
//:     "Full Skin": 7,
//:     "Full Standard": 8,
//:     "Full Tint": 9,
//:     "Glossiness": 10,
//:     "Mask 1": 11,
//:     "Mask 2": 12,
//:     "Mask 3": 13,
//:     "N Dot L": 14,
//:     "Normal": 15,
//:     "Reflection": 16,
//:     "Reflectivity": 17,
//:     "Solid Alpha": 18,
//:     "Specular": 19,
//:     "World Space Normal": 20
//:   }
//: }
uniform int i_technique;
//: param custom { "default": [1.0, 1.0, 1.0], "label": "Light Color", "widget": "color", "group": "Scene Lightning" }
uniform vec3 light_color0;
//: param custom { "default": 0, "label": "Relative Light Position X", "min": -100, "max": 100, "group": "Scene Lightning" }
uniform int light_relative0_x;
//: param custom { "default": 0, "label": "Relative Light Position Y", "min": -100, "max": 100, "group": "Scene Lightning" }
uniform int light_relative0_y;
//: param custom { "default": 0, "label": "Relative Light Position Z", "min": -100, "max": 100, "group": "Scene Lightning" }
uniform int light_relative0_z;
//: param custom { "default": false, "label": "Absolute Light Position", "group": "Scene Lightning" }
uniform bool light_absolute0;
//: param custom { "default": 10, "label": "Absolute Light Position X", "min": -100, "max": 100, "group": "Scene Lightning" }
uniform int light_absolute0_x;
//: param custom { "default": 20, "label": "Absolute Light Position Y", "min": -100, "max": 100, "group": "Scene Lightning" }
uniform int light_absolute0_y;
//: param custom { "default": 10, "label": "Absolute Light Position Z", "min": -100, "max": 100, "group": "Scene Lightning" }
uniform int light_absolute0_z;
//: param custom { "default": "Environment_Sharp", "label": "Reflection Map", "usage": "environment", "group": "Scene Lightning" }
uniform sampler2D s_environment_map;
//: param custom { "default": 1.0, "label": "Environment Exposure", "min": 0.0, "max": 10.0, "step": 0.1, "group": "Scene Lightning" }
uniform float f_environment_exposure;
//: param custom { "default": 0.0, "label": "Environment Rotation", "min": 0.0, "max": 1.0, "step": 0.01, "group": "Scene Lightning" }
uniform float f_environment_rotation;

// param custom { "default": true, "label": "Shadows", "group": "Scene Lightning" }
// uniform bool b_shadows;

//: param custom { "default": false, "label": "Enable Alpha" }
uniform bool b_enable_alpha;
//: param custom { "default": false, "label": "Use Opacity Channel" }
uniform bool b_use_opacity;
//: param custom { "default": 1, "label": "Alpha Mode", "min": 1, "max": 2 }
uniform int i_alpha_mode;
//: param custom { "default": 1.0, "label": "Height Force", "min": 0.01, "max": 10.0}
uniform float f_height_force;
//: param custom { "default": true, "label": "Faction Coloring" }
uniform bool b_faction_coloring;
//: param custom { "default": [0.5, 0.1, 0.1], "label": "Tint Color 1", "widget": "color" }
uniform vec3 vec3_color_0;
//: param custom { "default": [0.3, 0.6, 0.5], "label": "Tint Color 2", "widget": "color" }
uniform vec3 vec3_color_1;
//: param custom { "default": [0.5, 0.2, 0.1], "label": "Tint Color 3", "widget": "color" }
uniform vec3 vec3_color_2;
//: param custom { "default": false, "label": "Enable Decal" }
uniform bool b_do_decal;
//: param custom { "default": false, "label": "Enable Dirt", "group": "Dirt" }
uniform bool b_do_dirt;
//: param custom { "default": "default_decaldirtmap", "label": "Decal Dirtmap", "usage": "texture", "group": "Dirt" }
uniform sampler2D s_decal_dirtmap;
//: param custom { "default": true, "label": "Enable Offset U", "group": "Dirt" }
uniform bool b_random_tile_u;
//: param custom { "default": true, "label": "Enable Offset V", "group": "Dirt" }
uniform bool b_random_tile_v;
//: param custom { "default": 0.0, "label": "UV Offset U", "min": -1.0, "max": 1.0, "step": 0.01, "group": "Dirt" }
uniform float f_uv_offset_u;
//: param custom { "default": 0.0, "label": "UV Offset V", "min": -1.0, "max": 1.0, "step": 0.01, "group": "Dirt" }
uniform float f_uv_offset_v;
//: param custom { "default": 4.0, "label": "Tile Factor U", "min": -100.0, "max": 100.0, "step": 1.0, "group": "Dirt" }
uniform float f_uv2_tile_interval_u;
//: param custom { "default": 4.0, "label": "Tile Factor V", "min": -100.0, "max": 100.0, "step": 1.0, "group": "Dirt" }
uniform float f_uv2_tile_interval_v;
//: param custom { "default": "", "label": "Dirtmap", "usage": "texture", "group": "Dirt" }
uniform sampler2D s_dirtmap_uv2;

vec4 light_position0;

//: param auto channel_normal_is_set
uniform bool channel_normal_is_set;
//: param auto normal_blending_mode
uniform int normal_blending_mode;
//: param auto normal_y_coeff
uniform float base_normal_y_coeff;
//: param auto channel_height_size
uniform vec4 height_size; // width, height, width_inv, height_inv

const float pi = 3.14159265;
const float one_over_pi = 0.31830988618379067153776752674503;
const float real_approx_zero = 0.001;
const float texture_alpha_ref = 0.5;
const float HEIGHT_FACTOR = 400.0;

const vec4 vec4_uv_rect = vec4(0.0, 0.0, 1.0, 1.0);

//	Tone mapping parameters
const float Tone_Map_Black = 0.001;
const float Tone_Map_White = 10.0;
const float low_tones_scurve_bias = 0.33;
const float high_tones_scurve_bias = 0.66;

/////////////////////////
// Colorimetry Functions
/////////////////////////

float _linear(in float fGamma)
{
  return pow(max(fGamma, 0.0001), 2.2);
}

vec3 _linear(in vec3 vGamma)
{
	return pow(max(vGamma, 0.0001), vec3(2.2));
}

float _gamma(in float fLinear)
{
	return pow(max(fLinear, 0.0001), 1.0 / 2.2);
}

vec3 _gamma(in vec3 vLinear)
{
	return pow(max(vLinear, 0.0001), vec3(1.0 / 2.2));
}

float get_diffuse_scale_factor()
{
	return 0.004;
}

float get_game_hdr_lighting_multiplier()
{
	return 5000.0;
}

float get_luminance(in vec3 color)
{
	vec3 lumCoeff = vec3(0.299, 0.587, 0.114);
	float luminance = dot(color, lumCoeff);
	return clamp(luminance, 0.0, 1.0);
}

vec3 get_adjusted_faction_color(in vec3 color)
{
	vec3 fc = color;
	float lum = get_luminance(fc);
	float dark_scale = 1.5;
	float light_scale = 0.5;

	fc = fc * (mix(dark_scale, light_scale, lum));

	return fc;
}

/////////////////////////
// Conversion Functions
/////////////////////////

vec3 texcoordEnvSwizzle(in vec3 ref)
{
#ifdef FXCOMPOSER
    return vec3(ref.x, ref.y, -ref.z);
#else
	return -vec3(ref.x, -ref.z, ref.y);
#endif
}

vec3 normalSwizzle(in vec3 ref)
{
#ifdef FXCOMPOSER
	return ref.xyz;
#else
	return vec3(ref.y, ref.x, ref.z);
#endif
}

vec3 normalSwizzle_UPDATED(in vec3 ref)
{
#ifdef FXCOMPOSER
	return ref.xyz;
#else
  return vec3(ref.x, ref.z, ref.y);
#endif
}

float cos2sin(in float x)
{
	return sqrt(1.0 - x * x);
}

float cos2tan2(in float x)
{
	return (1.0 - x * x) / (x * x);
}

float contrast(inout float _val, in float _contrast)
{
	_val = ((_val - 0.5) * max(_contrast, 0.0)) + 0.5;
	return _val;
}

/////////////////////////
// Height To Normal
/////////////////////////

vec3 normalFromHeight(vec2 tex_coord, float height_force)
{
  // Normal computation using height map
  float h_r  = textureOffset(s_height_map, tex_coord, ivec2( 1,  0)).r;
  float h_l  = textureOffset(s_height_map, tex_coord, ivec2(-1,  0)).r;
  float h_t  = textureOffset(s_height_map, tex_coord, ivec2( 0,  1)).r;
  float h_b  = textureOffset(s_height_map, tex_coord, ivec2( 0, -1)).r;
  float h_rt = textureOffset(s_height_map, tex_coord, ivec2( 1,  1)).r;
  float h_lt = textureOffset(s_height_map, tex_coord, ivec2(-1,  1)).r;
  float h_rb = textureOffset(s_height_map, tex_coord, ivec2( 1, -1)).r;
  float h_lb = textureOffset(s_height_map, tex_coord, ivec2(-1, -1)).r;

  vec2 dh_dudv = (0.5 * -height_force) * height_size.xy * vec2(
    2.0 * (h_l - h_r) + h_lt - h_rt + h_lb - h_rb,
    2.0 * (h_b - h_t) + h_rb - h_rt + h_lb - h_lt);
  return normalize(vec3(dh_dudv, HEIGHT_FACTOR));
}

// Old normal + height blending

// vec3 normalBlendOriented(vec3 baseNormal, vec3 overNormal)
// {
//   baseNormal = (baseNormal * vec3(2.0)) + vec3(-1.0, -1.0, 0.0);
//   return normalize(baseNormal * dot(baseNormal, overNormal) - overNormal * baseNormal.z);
// }
//
// vec3 getTSNormal(vec2 tex_coord)
// {
//   vec3 baseNormal = texture(s_normal_tex, tex_coord).rgb;
//   baseNormal.y = 1.0 - baseNormal.y;
//
//   vec3 overNormal = normalFromHeight(tex_coord, f_height_force);
//
//   vec3 normal = normalSwizzle_UPDATED(normalBlendOriented(baseNormal, overNormal));
//   return normal;
// }

// New normal + normal + height blending with reoriented normal mapping
// http://blog.selfshadow.com/publications/blending-in-detail/

vec3 normalBlendOriented(vec3 baseNormal, vec3 overNormal)
{
  baseNormal = (baseNormal * vec3(2.0, 2.0, 2.0)) + vec3(-1.0, -1.0, 0.0);
  overNormal = (overNormal * vec3(-2.0, -2.0, 2.0)) + vec3(1.0, 1.0, -1.0);
  return normalize(baseNormal * dot(baseNormal, overNormal) - overNormal * baseNormal.z);
}

vec3 getTSNormal(vec2 tex_coord)
{
  // Normal texture
  vec3 baseNormal = texture(s_normal_tex, tex_coord).rgb;
  baseNormal.y = 1.0 - baseNormal.y;

  // Normal channel
  vec3 overNormal1 = texture(s_normal_map, tex_coord).rgb;

  // Height channel
  vec3 overNormal2 = normalFromHeight(tex_coord, f_height_force);
  overNormal2 = (overNormal2 - vec3(1.0, 1.0, -1.0)) / vec3(-2.0, -2.0, 2.0);

  vec3 normal = normalBlendOriented(baseNormal, overNormal1);
  normal = (normal - vec3(1.0, 1.0, -1.0)) / vec3(-2.0, -2.0, 2.0);
  normal.r = 1.0 - normal.r;
  normal.g = 1.0 - normal.g;
  return normalSwizzle_UPDATED(normalBlendOriented(normal, overNormal2));
}

/////////////////////////
// Forward Declarations
/////////////////////////

vec3 tone_map_linear_hdr_pixel_value(in vec3 linear_hdr_pixel_val);
vec4 HDR_RGB_To_HDR_CIE_Log_Y_xy(in vec3 linear_color_val);
vec4 tone_map_HDR_CIE_Log_Y_xy_To_LDR_CIE_Yxy(in vec4 hdr_LogYxy);
vec4 LDR_CIE_Yxy_To_Linear_LDR_RGB(in vec4 ldr_cie_Yxy);
float get_scurve_y_pos(const float x_coord);

/////////////////////////
// Lighting Functions
/////////////////////////

vec3 rotate(vec3 v, float a)
{
	float angle = a * 2.0 * pi;
	float ca = cos(angle);
	float sa = sin(angle);
	return vec3(v.x * ca + v.z * sa, v.y, v.z * ca - v.x * sa);
}

vec3 get_environment_color_UPDATED(in vec3 direction , in float lod)
{
  vec2 pos = one_over_pi * vec2(atan(-direction.z, -1.0 * direction.x), 2.0 * asin(direction.y));
  pos = (0.5 * pos) + vec2(0.5);
  return _linear(textureLod(s_environment_map, pos, lod).rgb) * f_environment_exposure;
}

vec3 cube_ambient(in vec3 N)
{
  // return _linear(texture(s_ambient, texcoordEnvSwizzle(N).rg).rgb);
  return vec3(0.0, 0.0, 0.0);
}

vec2 phong_diffuse(in vec3 N, in vec3 L)
{
	float factor = max(0.0, dot(N, -L));
	return vec2(factor, (factor > 0.0));
}

float phong_specular(in vec3 I, in vec3 N, in float shininess, in vec3 L)
{
	vec3 R = reflect(L, N);
  return clamp(pow(max(0.0, dot(R, -I)), shininess), 0.0, 1.0);
}

float aniso_specular(in vec3 I, in vec3 N, in vec3 T, in float shininess, in vec3 L)
{
  vec3 nH = normalize(I + L);
  vec3 nT = normalize(T);
	nT = normalize(nT - N * dot(N, nT));
  float spec = pow(sqrt(1.0 - (pow(dot(nT, nH), 2.0))), shininess);
  return spec;
}

float blinn_specular(in vec3 I, in vec3 N, in float shininess, in vec3 L)
{
  shininess = shininess * 4.0;
  vec3 H = normalize(I + L);
	vec3 R = reflect(L, N);
  return clamp(pow(max(0.0, dot(N, -H)), shininess), 0.0, 1.0);
}

float blinn_phong_specular(in float dotNH, in float SpecularExponent)
{
  float D = pow(dotNH, SpecularExponent) * (SpecularExponent + 1.0) / 2.0;
  return D;
}

/////////////////////////
// Cook Torrance Model
/////////////////////////

float beckmann_distribution(in float dotNH, in float SpecularExponent)
{
    float invm2 = SpecularExponent / 2.0;
    float D = exp(-cos2tan2(dotNH) * invm2) / pow(dotNH, 4.0) * invm2;
    return D;
}

vec3 fresnel_optimized(in vec3 R, in float c)
{
  vec3 F = mix(R, clamp(60.0 * R, 0.0, 1.0), pow(1.0 - c, 4.0));
  return F;
}

vec3 fresnel_full(in vec3 R, in float c)
{
    vec3 n = (1.0 + sqrt(R)) / (1.0 - sqrt(R));
    vec3 FS = (c - n * sqrt(1.0 - pow(cos2sin(c) / n, vec3(2.0, 2.0, 2.0)))) / (c + n * sqrt(1.0 - pow(cos2sin(c) / n, vec3(2.0, 2.0, 2.0))));
    vec3 FP = (sqrt(1.0 - pow(cos2sin(c) / n, vec3(2.0, 2.0, 2.0))) - n * c) / (sqrt(1.0 - pow(cos2sin(c) / n, vec3(2.0, 2.0, 2.0))) + n * c);
    return (FS * FS + FP * FP) / 2.0;
}

/////////////////////////
// Decal / Dirt
/////////////////////////

void ps_common_blend_decal(in vec4 color, in vec3 normal, in vec3 specular, in float reflectivity, out vec4 ocolor, out vec3 onormal, out vec3 ospecular, out float oreflectivity, in vec2 uv, in float decal_index, in vec4 uv_rect_coords, in float valpha)
{
	vec2 decal_top_left = uv_rect_coords.xy;
	vec2 decal_dimensions = uv_rect_coords.zw - uv_rect_coords.xy;

	vec2 decal_uv = (uv-decal_top_left) / decal_dimensions;

	vec4 decal_diffuse;
	vec3 decal_normal;

	decal_diffuse = texture(s_decal_diffuse, decal_uv).rgba;
  decal_diffuse.rgb = _linear(decal_diffuse.rgb);
  vec3 dNp = texture(s_decal_normal, decal_uv).rgb;
	decal_normal = normalSwizzle_UPDATED((dNp.rgb * 2.0) - 1.0);
	float decal_mask = texture(s_decal_mask, uv).r;

	float decalblend = decal_mask * decal_diffuse.a * valpha;
	oreflectivity = mix(reflectivity, reflectivity * 0.5, decalblend);
	ocolor = vec4(1.0);
	onormal = vec3(0.0, 0.0, 1.0);
	ospecular = mix(specular, decal_diffuse.rgb, decalblend);

	ocolor.rgb = mix(color.rgb, decal_diffuse.rgb, decalblend);

	onormal.xyz = mix(onormal.xyz, decal_normal.rgb, decalblend);
	onormal.xyz = vec3(normal.xy + onormal.xy, normal.z);
}

void ps_common_blend_dirtmap(inout vec4 color, inout vec3 normal, in vec3 specular, inout float reflectivity, out vec4 ocolor, out vec3 onormal, out vec3 ospecular, out float oreflectivity, in vec2 uv, in vec2 uv_offset)
{
  if (b_random_tile_u)
  {
    if (b_random_tile_v)
    {
      uv_offset = uv_offset * vec2(1, 1);
    } else
    {
      uv_offset = uv_offset * vec2(1, 0);
    }
  } else
  {
    if (b_random_tile_v)
    {
      uv_offset = uv_offset * vec2(0, 1);
    } else
    {
      uv_offset = uv_offset * vec2(0, 0);
    }
  }

	float mask_alpha = texture(s_decal_dirtmask, uv).r;
	vec4 dirtmap = texture(s_decal_dirtmap, (uv + uv_offset) * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));
  dirtmap.g = 1.0 - dirtmap.g;

	float d_strength = 1.0;

	vec2 dirt_normal = (vec2(dirtmap.r, dirtmap.g) * 2.0) - 1.0;

	float dirt_alpha = dirtmap.a;
	float dirt_alpha_blend = mask_alpha * dirt_alpha * d_strength;

	vec3 dirt_color = vec3(0.03, 0.03, 0.02);
	ocolor = color;
	onormal = normal;

	ocolor.rgb = mix(color.rgb, dirt_color, dirt_alpha_blend);

	ospecular = mix(specular, dirt_color, dirt_alpha_blend);

	onormal.xz += (dirt_normal.xy * mask_alpha * d_strength);
	onormal = normalize(onormal);

	oreflectivity = reflectivity;
}

void ps_common_blend_vfx(inout vec4 color, inout vec3 normal, in vec3 specular, inout float reflectivity, out vec4 ocolor, out vec3 onormal, out vec3 ospecular, out float oreflectivity, in vec2 uv, in vec2 uv_offset)
{
  if (b_random_tile_u)
  {
    if (b_random_tile_v)
    {
      uv_offset = uv_offset * vec2(1, 1);
    } else
    {
      uv_offset = uv_offset * vec2(1, 0);
    }
  } else
  {
    if (b_random_tile_v)
    {
      uv_offset = uv_offset * vec2(0, 1);
    } else
    {
      uv_offset = uv_offset * vec2(0, 0);
    }
  }

	vec4 dirtmap = texture(s_decal_dirtmap, (uv + uv_offset) * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));

	ocolor = vec4(mix(color.rgb, dirtmap.rgb, dirtmap.a), 1.0);

	onormal = normal;
	ospecular = specular;
	oreflectivity = reflectivity;
}

/////////////////////////
// Skin Lightning Model
/////////////////////////

struct SkinLightingModelMaterial
{
  float Gloss;
  float SpecularLevel;
	float RimMask;
  float SubSurfaceStrength;
	float BackScatterStrength;
  vec3 Color;
	vec3 specular_color;
	vec3 Normal;
	float Depth;
	float Shadow;
	float SSAO;
};

SkinLightingModelMaterial create_skin_lighting_material(in vec2 _MaterialMap, in vec3 _SkinMap, in vec3 _Color, in vec3 _specular_color, in vec3 _Normal, in vec4 _worldposition)
{
	SkinLightingModelMaterial material;

	material.Gloss = _MaterialMap.x;
	material.SpecularLevel = _MaterialMap.y;
	material.RimMask = _SkinMap.x;
	material.SubSurfaceStrength = _SkinMap.y;
	material.BackScatterStrength = _SkinMap.z;
	material.Color = _Color;
	material.specular_color = _specular_color;
	material.Normal = normalize(_Normal);
	material.Depth = 1.0;
	material.Shadow = 1.0;
	material.SSAO = 1.0;

	return material;
}

vec3 skin_shading(in vec3 L, in vec3 N, in vec3 V, in float sss_strength, in vec3 color1, in vec3 color2)
{
	float ndotl = dot(N, -L);

	vec3 diff1 = vec3(ndotl * clamp(((ndotl * 0.8) + 0.3) / 1.44, 0.0, 1.0));
	vec3 diff2 = color1 * (clamp(((ndotl * 0.9) + 0.5) / 1.44, 0.0, 1.0)) * clamp(1.0 - (diff1 + 0.3), 0.0, 1.0);
	vec3 diff3 = color2 * (clamp(((ndotl * 0.3) + 0.3) / 2.25, 0.0, 1.0)) * (1.0 - diff1) * (1.0 - diff2);

	vec3 mixDiff = diff1 + diff2 + diff3;

	vec3 blendedDiff = mix(vec3(ndotl), mixDiff, sss_strength);
	return clamp(vec3(blendedDiff), 0.0, 1.0);
}

float get_skin_dlight_diffuse_scaler()
{
  return 0.9;
}

float get_skin_dlight_specular_scaler()
{
  return 2.0;
}

float get_skin_dlight_rim_scaler()
{
  return 1.0;
}

vec3 skin_lighting_model_directional_light(in vec3 LightColor, in vec3 normalised_light_dir, in vec3 normalised_view_dir, in SkinLightingModelMaterial skinlm_material)
{
	LightColor *= get_game_hdr_lighting_multiplier();

	normalised_light_dir = -normalised_light_dir;

  vec3 diffuse_scale_factor = vec3(get_diffuse_scale_factor());

	float normal_dot_light_dir = max(dot(skinlm_material.Normal, -normalised_light_dir), 0.0);

	vec3 dlight_diffuse = skinlm_material.Color.rgb * skin_shading(normalised_light_dir, skinlm_material.Normal, normalised_view_dir, skinlm_material.SubSurfaceStrength, vec3(0.612066, 0.456263, 0.05), vec3(0.32, 0.05, 0.006)) * LightColor * diffuse_scale_factor;
	dlight_diffuse *= get_skin_dlight_diffuse_scaler();

	float backscatter = pow(clamp(dot(skinlm_material.Normal, normalised_light_dir), 0.0, 1.0), 2.0) * pow(clamp(dot(normalised_view_dir, -normalised_light_dir), 0.0, 1.0), 4.0) * skinlm_material.BackScatterStrength;

	vec3 backscatter_color = mix(LightColor, LightColor * vec3(0.7, 0.0, 0.0), skinlm_material.SubSurfaceStrength) * diffuse_scale_factor * backscatter * skinlm_material.Shadow;

	dlight_diffuse += (skinlm_material.Color.rgb * backscatter_color * get_skin_dlight_diffuse_scaler());

	vec3  env_light_diffuse = skinlm_material.Color.rgb * cube_ambient(skinlm_material.Normal).rgb;

  float kspec = phong_specular(normalised_view_dir, skinlm_material.Normal, mix(1.0, 128.0, (skinlm_material.Gloss * skinlm_material.Gloss)), normalised_light_dir );

	vec3 dlight_specular = skinlm_material.SpecularLevel * kspec * LightColor * diffuse_scale_factor * get_skin_dlight_specular_scaler();

  vec3 reflected_view_vec = reflect(-normalised_view_dir, skinlm_material.Normal);

 vec3 rim_env_color = cube_ambient(reflected_view_vec).rgb;

	float rimfresnel = clamp(1.0 - dot(-normalised_view_dir, skinlm_material.Normal), 0.0, 1.0);

  //1.5 is a scalar in order that we can get a brighter rim by setting the texture to be white. Previously the maximum brightness was quite dull.
	float riml = clamp(pow(rimfresnel, 2.0), 0.0, 1.0) * skinlm_material.RimMask * 1.5 * get_skin_dlight_rim_scaler();

	float upness = max(dot(normalize(skinlm_material.Normal + vec3(0.0, 0.75, 0.0)), vec3(0.0, 1.0, 0.0)), 0.0);

  vec3  env_light_specular = (riml * upness * rim_env_color);

	float   shadow_attenuation      = skinlm_material.Shadow;

  return (skinlm_material.SSAO * (env_light_diffuse + env_light_specular)) + (shadow_attenuation * (dlight_specular + dlight_diffuse));
}

/////////////////////////
// Standard Lightning Model
/////////////////////////

struct StandardLightingModelMaterial_UPDATED
{
	vec3 Diffuse_Color;
  vec3 Specular_Color;
	vec3 Normal;
	float Smoothness;
  float Reflectivity;
	float Depth;
	float Shadow;
	float SSAO;
};

StandardLightingModelMaterial_UPDATED create_standard_lighting_material_UPDATED(in vec3 _DiffuseColor, in vec3 _Specular_Color, in vec3 _Normal, in float _Smoothness, in float _Reflectivity, in vec4 _worldposition)
{
  StandardLightingModelMaterial_UPDATED material;

  material.Diffuse_Color = _DiffuseColor;
  material.Specular_Color = _Specular_Color;
  material.Normal = _Normal;
  material.Smoothness = _Smoothness;
  material.Reflectivity = _Reflectivity;
  material.Depth = 1.0;
  material.Shadow = 1.0;
  material.SSAO = 1.0;

  return material;
}

float get_sun_angular_diameter()
{
  const float suns_angular_diameter = 1.0;

  return radians(suns_angular_diameter);
}

float get_sun_angular_radius()
{
  return 0.5 * get_sun_angular_diameter();
}

float get_error_func_a_value()
{
  return (8.0 * (pi - 3.0)) / ( 3.0 * pi * (4.0 - pi));
}

float erf(in float x)
{
  float x_squared = x * x;

  float a = get_error_func_a_value();

  float a_times_x_squared = a * x_squared;

  float numerator = ( 4.0 * one_over_pi ) + ( a_times_x_squared );

  float denominator = 1.0 + a_times_x_squared;

  float main_term = -1.0 * x_squared * ( numerator / denominator );

  return sign(x) * sqrt(1.0 - exp(main_term));
}

float erfinv(in float x)
{
  float one_over_a = 1.0 / get_error_func_a_value();

  float log_1_minus_x_squared = log(1.0 - (x * x));

  float root_of_first_term = (2.0 * one_over_pi * one_over_a) + (log_1_minus_x_squared * 0.5);

  float first_term = root_of_first_term * root_of_first_term;

  float second_term = log_1_minus_x_squared * one_over_a;

  float third_term = (2.0 * one_over_pi * one_over_a) + (log_1_minus_x_squared * 0.5);

  float all_terms = first_term - second_term - third_term;

  return sign(x) * sqrt(sqrt(first_term - second_term) - third_term);
}

float norm_cdf(in float x, in float sigma)
{
  float one_over_root_two = 0.70710678118654752440084436210485;

  return 0.5 * (1.0 + erf(x * (1.0 / sigma) * one_over_root_two));
}

float norm_cdf(in float x_min, in float x_max, in float sigma)
{
  float min_summed_area = norm_cdf(x_min, sigma);
  float max_summed_area = norm_cdf(x_max, sigma);

  return max_summed_area - min_summed_area;
}

float norminv_sigma(in float x, in float area_under_the_graph)
{
  float error_func_x = erfinv((2.0 * area_under_the_graph) - 1.0);

  float sigma = x / (error_func_x * 1.4142135623730950488016887242097);

  return sigma;
}

float get_normal_distribution_sigma_about_origin(in float area_under_graph_centred_around_origin, in float x_half_distance_from_origin)
{
  float area_from_x_neg_infinity = 0.5 + (0.5 * area_under_graph_centred_around_origin);

  return norminv_sigma(x_half_distance_from_origin, area_from_x_neg_infinity);
}

float determine_fraction_of_facets_at_reflection_angle(in float smoothness, in float light_vec_reflected_view_vec_angle)
{
  float sun_angular_radius = get_sun_angular_radius();

  float max_fraction_of_facets = 0.9999;
  float min_fraction_of_facets = get_diffuse_scale_factor();

  float fraction_of_facets = mix(min_fraction_of_facets, max_fraction_of_facets, smoothness * smoothness);

  float fraction_of_facets_to_look_for = 0.5 + (fraction_of_facets * 0.5);

  float sigma = max(norminv_sigma(sun_angular_radius, fraction_of_facets_to_look_for), 0.0001);

  float proportion_of_facets = norm_cdf(light_vec_reflected_view_vec_angle - sun_angular_radius, light_vec_reflected_view_vec_angle + sun_angular_radius, sigma);

  return proportion_of_facets;
}

float determine_facet_visibility(in float roughness, in vec3 normal_vec, in vec3 light_vec)
{
	float	n_dot_l = clamp(dot(normal_vec, light_vec), 0.0, 1.0);
	float	towards_diffuse_color_surface	= sin(roughness * pi * 0.5); //	( 0 - 1 ) output...
	float	facet_visibility = mix(1.0, n_dot_l, towards_diffuse_color_surface);

	return facet_visibility;
}

vec3 determine_surface_reflectivity(in vec3 material_reflectivity, in float roughness, in vec3 light_vec, in vec3 view_vec)
{
	float fresnel_curve = 10.0;

  float val1 = max(0.0, dot(light_vec, -view_vec));

  float val2 = pow(val1, fresnel_curve);

	float	fresnel_bias	= 0.5;

	float	roughness_bias = 0.98;

	float	smoothness_val = pow(cos(roughness * roughness_bias * pi * 0.5), fresnel_bias);

  return mix(material_reflectivity, clamp(60.0 * material_reflectivity, 0.0, 1.0), val2 * smoothness_val);
}

vec4 plot_standard_lighting_model_test_func(in vec4 vpos)
{
	vec4 g_vpos_texel_offset = vec4(0.0);
	vec4 g_screen_size_minus_one = vec4(0.0);

  vpos -= g_vpos_texel_offset.xxxx;

  float xpos = vpos.x / g_screen_size_minus_one.x * 5.0;
  float ypos = ((g_screen_size_minus_one.y - vpos.y) / g_screen_size_minus_one.y) * 10.0;

  float y_value = norminv_sigma(mix(0.01, 5.0, xpos), 0.7);

  return vec4(clamp((y_value * g_screen_size_minus_one.y) - (ypos * g_screen_size_minus_one.y), 0.0, 1.0));
}

vec3 get_reflectivity_base(in vec3 light_vec, in vec3 normal_vec, in vec3 view_vec , in vec3 material_reflectivity , in float smoothness , in float roughness , in float light_vec_reflected_view_vec_angle)
{
  float n_dot_l	= dot(light_vec, normal_vec);

	if (n_dot_l <= 0.0)
  {
    return vec3(0.0);
  }

	float fraction_of_facets = determine_fraction_of_facets_at_reflection_angle(smoothness, light_vec_reflected_view_vec_angle);

  float facet_visibility = determine_facet_visibility(roughness, normal_vec, light_vec);  // Looks ok

  vec3 surface_reflectivity = determine_surface_reflectivity(material_reflectivity, roughness, light_vec, view_vec);

  return fraction_of_facets * facet_visibility * surface_reflectivity;
}

float ensure_correct_trig_value(in float value)
{
	return clamp(value, -1.0, +1.0);
}

vec3 get_reflectivity_dir_light(in vec3 light_vec, in vec3 normal_vec, in vec3 view_vec, in vec3 reflected_view_vec, in float standard_material_reflectivity, in float smoothness, in float roughness)
{
  float light_vec_reflected_view_vec_angle = acos(ensure_correct_trig_value(dot(light_vec, reflected_view_vec)));

  return get_reflectivity_base(light_vec, normal_vec, view_vec, vec3(standard_material_reflectivity), smoothness, roughness, light_vec_reflected_view_vec_angle);
}

vec3 get_reflectivity_env_light(in vec3 light_vec, in vec3 normal_vec, in vec3 view_vec, in float standard_material_reflectivity, in float smoothness, in float roughness)
{
	return determine_surface_reflectivity(vec3(standard_material_reflectivity), roughness, light_vec, view_vec);
}

vec3 standard_lighting_model_directional_light_UPDATED(in vec3 LightColor, in vec3 normalised_light_dir, in vec3 normalised_view_dir, in StandardLightingModelMaterial_UPDATED material)
{
  LightColor *= get_game_hdr_lighting_multiplier();
  vec3 diffuse_scale_factor = vec3(get_diffuse_scale_factor());

  float roughness = 1.0 - material.Smoothness;

	normalised_light_dir = normalised_light_dir;
	normalised_view_dir = -normalised_view_dir;

	float normal_dot_light_vec = max(0.0, dot(material.Normal, normalised_light_dir));

  vec3 reflected_view_vec = reflect(-normalised_view_dir, material.Normal);

  float texture_num_lods = 10.0;

	float env_map_lod = roughness * (texture_num_lods - 1.0);

  vec3 environment_color = get_environment_color_UPDATED(rotate(reflected_view_vec, f_environment_rotation), env_map_lod);

  vec3 dlight_pixel_reflectivity = get_reflectivity_dir_light(normalised_light_dir, material.Normal, normalised_view_dir, reflected_view_vec, material.Reflectivity, material.Smoothness, roughness);
  vec3 dlight_specular_color = dlight_pixel_reflectivity * material.Specular_Color * LightColor;
  // All photons not accounted for by reflectivity are accounted by scattering. From the energy difference between incoming light and emitted light we could calculate the amount of energy turned into heat. This energy would not be enough to make a viewable difference at standard illumination levels.
  vec3 dlight_material_scattering  = 1.0 - max(dlight_pixel_reflectivity, material.Reflectivity);

  vec3  env_light_pixel_reflectivity = max(vec3(material.Reflectivity), get_reflectivity_env_light(reflected_view_vec, material.Normal, normalised_view_dir, material.Reflectivity, material.Smoothness, roughness));
  vec3  env_light_specular_color = environment_color * env_light_pixel_reflectivity * material.Specular_Color;

	vec3  dlight_diffuse = material.Diffuse_Color * normal_dot_light_vec * LightColor * dlight_material_scattering;

	vec3 ambient_color = cube_ambient(material.Normal);

  vec3 env_light_diffuse = ambient_color * material.Diffuse_Color * (1.0 - material.Reflectivity);

  dlight_diffuse *= diffuse_scale_factor;

	// if (!b_shadows)
	// {
	// 	material.Shadow = 1.0;
	// }
	// else
	// {
	// 	material.Shadow = material.Shadow;
	// }
	float shadow_attenuation = material.Shadow;
	float reflection_shadow_attenuation = (1.0 - ((1.0 - material.Shadow) * 0.75));

  return (material.SSAO * (env_light_diffuse + (reflection_shadow_attenuation * env_light_specular_color))) + (shadow_attenuation * (dlight_specular_color + dlight_diffuse));
}

float check_alpha(in float pixel_alpha)
{
  // Alpha Enabled
	if (b_enable_alpha)
	{
    // Alpha Test
    if (i_alpha_mode == 1)
    {
      if (pixel_alpha - texture_alpha_ref < 0.0)
      {
        discard;
      } else
      {
        return 1.0;
      }
    }
    // Alpha Blend
    else
    {
      return pixel_alpha;
    }
	}
  // Alpha Disabled
  else
  {
    return 1.0;
  }
}

/////////////////////////
// Tone Mapper
/////////////////////////

vec4 HDR_RGB_To_HDR_CIE_Log_Y_xy(in vec3 linear_color_val)
{
	mat3 cie_transform_mat = mat3(0.4124, 0.3576, 0.1805,
										            0.2126, 0.7152, 0.0722,
										            0.0193, 0.1192, 0.9505);

	vec3 cie_XYZ = cie_transform_mat * linear_color_val;

	float denominator = cie_XYZ.x + cie_XYZ.y + cie_XYZ.z;

	float x = cie_XYZ.x / max(denominator, real_approx_zero);
	float y = cie_XYZ.y / max(denominator, real_approx_zero);

  return vec4(log(max(cie_XYZ.y, real_approx_zero)) / log(10.0), x, y, cie_XYZ.y);
}

float get_scurve_y_pos(const float x_coord)
{
  float point0_y = 0.0;
  float point1_y = low_tones_scurve_bias;
  float point2_y = high_tones_scurve_bias;
  float point3_y = 1.0;

  vec4 t = vec4(x_coord * x_coord * x_coord, x_coord * x_coord, x_coord, 1.0);

  mat4 BASIS = mat4(-1.0,	+3.0,	-3.0,	+1.0,
        			      +3.0,	-6.0,	+3.0,	+0.0,
        			      -3.0,	+3.0,	+0.0,	+0.0,
                    +1.0,	+0.0,	+0.0,	+0.0);

  vec4 g = BASIS * t;


  return (point0_y * g.x) + (point1_y * g.y) + (point2_y * g.z) + (point3_y * g.w);
}

vec4 tone_map_HDR_CIE_Log_Y_xy_To_LDR_CIE_Yxy(in vec4 hdr_LogYxy)
{
  float black_point = Tone_Map_Black;
  float white_point = Tone_Map_White;
  float log_Y_black_point = log(Tone_Map_Black) / log(10.0);
  float log_Y_white_point = log(Tone_Map_White) / log(10.0);

  hdr_LogYxy.x = max(hdr_LogYxy.x, log_Y_black_point);

  float log_y_display_range = log_Y_white_point - log_Y_black_point;

	float log_y_in_white_black = (hdr_LogYxy.x - log_Y_black_point) / log_y_display_range;

  float log_y_in_white_black_scurve_biased = get_scurve_y_pos(log_y_in_white_black);

  float biased_log_y = log_Y_black_point + (log_y_in_white_black_scurve_biased * log_y_display_range);

  float biased_y = pow(10.0, biased_log_y);

  float ldr_y = (biased_y - black_point) / (white_point - black_point);

  return vec4(ldr_y, hdr_LogYxy.yzw);
}

vec4 LDR_CIE_Yxy_To_Linear_LDR_RGB(in vec4 ldr_cie_Yxy)
{
  float Y = ldr_cie_Yxy[0];
  float x = ldr_cie_Yxy[1];
  float y = ldr_cie_Yxy[2];

	float	safe_denominator = max(y, real_approx_zero);

  float cie_Y = Y;

  vec3 cie_XYZ = vec3(x * cie_Y / safe_denominator, cie_Y, (1.0 - x - y) * cie_Y / safe_denominator);


	mat3 cie_XYZ_toRGB_transform_mat = mat3(+3.2405, -1.5372, -0.4985,
										                      -0.9693, +1.8760, +0.0416,
										                      +0.0556, -0.2040, +1.0572);

  vec3 rgb = cie_XYZ_toRGB_transform_mat * cie_XYZ;

	rgb.xyz = max(vec3(0.0, 0.0, 0.0), rgb);

  return vec4(rgb.xyz, 1.0);
}

vec3 tone_map_linear_hdr_pixel_value(in vec3 linear_hdr_pixel_val)
{
	vec4 hdr_CIE_LogYxy_pixel = HDR_RGB_To_HDR_CIE_Log_Y_xy(linear_hdr_pixel_val);

  vec4 tone_mapped_ldr_CIE_Yxy_pixel = tone_map_HDR_CIE_Log_Y_xy_To_LDR_CIE_Yxy(hdr_CIE_LogYxy_pixel);

  vec4 tone_mapped_ldr_linear_rgb = LDR_CIE_Yxy_To_Linear_LDR_RGB(tone_mapped_ldr_CIE_Yxy_pixel);

  return tone_mapped_ldr_linear_rgb.rgb;
}

// Shader entry point
vec4 shade(V2F inputs)
{
  mat3 basis = mat3(normalize(inputs.tangent), normalize(inputs.normal), normalize(inputs.bitangent));

  if (i_technique == 0) // Ambient Occlusion
  {
    vec3 ao_tex = texture(s_ambient_occlusion_tex, inputs.tex_coord.xy).rgb;
    vec3 ao = texture(s_ambient_occlusion, inputs.tex_coord.xy).rgb;

    return vec4(_linear(ao_tex.rrr * ao.rrr), 1.0);
  } else if (i_technique == 1) // Vertex Color
  {
    vec4 Ct = texture(s_diffuse_color, inputs.tex_coord.xy);

    return vec4(inputs.color[0].rgb, Ct.a);
  } else if (i_technique == 2) // Custom Terrain
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

    vec3 light_vector = normalize(light_position0.xyz - inputs.position);

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    vec4 specular_color = texture(s_specular_color, inputs.tex_coord.xy);

    float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

    smoothness = _linear(smoothness);

    float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

    reflectivity = _linear(reflectivity);

    vec3 N = getTSNormal(inputs.tex_coord.xy);

    ps_common_blend_decal(diffuse_color, N, specular_color.rgb, reflectivity, diffuse_color, N, specular_color.rgb, reflectivity, inputs.tex_coord.xy, 0, vec4_uv_rect, 1 - inputs.color[0].a);

    vec3 pixel_normal = normalize(basis * normalize(N));

    StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

    vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

    vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

    return vec4(ldr_linear_col, alpha);
  } else if (i_technique == 3) // Decal Dirt
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

  	vec3 light_vector = normalize(light_position0.xyz - inputs.position);

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    vec4 specular_color = texture(s_specular_color, inputs.tex_coord.xy);

    float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

    smoothness = _linear(smoothness);

    float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

    reflectivity = _linear(reflectivity);

  	float mask_p1 = texture(s_mask1, inputs.tex_coord.xy).r;
  	float mask_p2 = texture(s_mask2, inputs.tex_coord.xy).r;
  	float	mask_p3 = texture(s_mask3, inputs.tex_coord.xy).r;

  	diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_0.rgb), mask_p1);
  	diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_1.rgb), mask_p2);
  	diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_2.rgb), mask_p3);

    vec3 N = getTSNormal(inputs.tex_coord.xy);

  	if (b_do_decal)
  	{
  		ps_common_blend_decal(diffuse_color, N, specular_color.rgb, reflectivity, diffuse_color, N, specular_color.rgb, reflectivity, inputs.tex_coord.xy, 0, vec4_uv_rect, 1.0);
  	}

  	if (b_do_dirt)
  	{
  		ps_common_blend_dirtmap(diffuse_color, N, specular_color.rgb, reflectivity, diffuse_color, N, specular_color.rgb, reflectivity, inputs.tex_coord.xy, vec2(f_uv_offset_u, f_uv_offset_v));
  	}

  	vec3 pixel_normal = normalize(basis * normalize(N));

    StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

    vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

    vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

    return vec4(ldr_linear_col, alpha);
  } else if (i_technique == 4) // Diffuse
  {
    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    return vec4(diffuse_color.rgb, alpha);
  } else if (i_technique == 5) // Full Ambient Occlusion
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

  	vec3 light_vector = normalize(light_position0.xyz - inputs.position);

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    vec4 specular_color = texture(s_specular_color, inputs.tex_coord.xy);

    float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

    smoothness = _linear(smoothness);

    float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

    reflectivity = _linear(reflectivity);

  	vec3 ao_tex = texture(s_ambient_occlusion_tex, inputs.tex_coord.xy).rgb;
    vec3 ao = texture(s_ambient_occlusion, inputs.tex_coord.xy).rgb;
  	diffuse_color.rgb *= ao_tex.rrr * ao.rrr;
  	specular_color.rgb *= ao_tex.rrr * ao.rrr;

  	vec3 N = getTSNormal(inputs.tex_coord.xy);
  	vec3 pixel_normal = normalize(basis * normalize(N));

    StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

    vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

    vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

    return vec4(ldr_linear_col, alpha);
  } else if (i_technique == 6) // Full Dirtmap
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

  	vec3 light_vector = normalize(light_position0.xyz - inputs.position);

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    vec4 specular_color = texture(s_specular_color, inputs.tex_coord.xy);

    float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

    smoothness = _linear(smoothness);

    float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

    reflectivity = _linear(reflectivity);

  	vec4 dirtmap = texture(s_dirtmap_uv2, inputs.tex_coord.xy * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));
    float alpha_mask = texture(s_alpha_mask, inputs.tex_coord.xy).r;

    float blend_amount = alpha_mask;

    float hardness = 1.0;

    float blend_2 = blend_amount * mix(1.0, dirtmap.a, blend_amount);

    blend_amount = clamp(((blend_2 - 0.5) * hardness) + 0.5, 0.0, 1.0);

    diffuse_color.rgb = diffuse_color.rgb * (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));
  	specular_color.rgb *= (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));

  	vec3 N = getTSNormal(inputs.tex_coord.xy);
  	vec3 pixel_normal = normalize(basis * normalize(N));

    StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

    vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

    vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

    return vec4(ldr_linear_col, alpha);
  } else if (i_technique == 7) // Full Skin
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

  	vec3 light_vector = normalize(light_position0.xyz - inputs.position);

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    vec4 specular_color = texture(s_specular_color, inputs.tex_coord.xy);

    float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

    smoothness = _linear(smoothness);

    float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

    reflectivity = _linear(reflectivity);

  	float mask_p1 = texture(s_mask1, inputs.tex_coord.xy).r;
  	float mask_p2 = texture(s_mask2, inputs.tex_coord.xy).r;
  	float mask_p3 = texture(s_mask3, inputs.tex_coord.xy).r;

  	vec3 N = getTSNormal(inputs.tex_coord.xy);

  	if (b_do_decal)
  	{
  		ps_common_blend_decal(diffuse_color, N, specular_color.rgb, reflectivity, diffuse_color, N, specular_color.rgb, reflectivity, inputs.tex_coord.xy, 0.0, vec4_uv_rect, 1.0);
  	}

  	if (b_do_dirt)
  	{
  		ps_common_blend_dirtmap(diffuse_color, N, specular_color.rgb, reflectivity, diffuse_color, N, specular_color.rgb, reflectivity, inputs.tex_coord.xy, vec2(f_uv_offset_u, f_uv_offset_v));
  	}

  	vec3 pixel_normal = normalize(basis * normalize(N));

    SkinLightingModelMaterial skin_mat = create_skin_lighting_material(vec2(smoothness, reflectivity), vec3(mask_p1, mask_p2, mask_p3), diffuse_color.rgb, specular_color.rgb, pixel_normal, vec4(inputs.position.xyz, 1.0));

    vec3 hdr_linear_col = skin_lighting_model_directional_light(light_color0, light_vector, eye_vector, skin_mat);

    vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

    return vec4(ldr_linear_col, alpha);
  } else if (i_technique == 8) // Full Standard
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

  	vec3 light_vector = normalize(light_position0.xyz - inputs.position);

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    vec4 specular_color = texture(s_specular_color, inputs.tex_coord.xy);

    float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

    smoothness = _linear(smoothness);

    float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

    reflectivity = _linear(reflectivity);

  	float mask_p1 = texture(s_mask1, inputs.tex_coord.xy).r;
  	float mask_p2 = texture(s_mask2, inputs.tex_coord.xy).r;
  	float mask_p3 = texture(s_mask3, inputs.tex_coord.xy).r;


  	if (b_faction_coloring)
  	{
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * get_adjusted_faction_color(_linear(vec3_color_0.rgb)), mask_p1);
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * get_adjusted_faction_color(_linear(vec3_color_1.rgb)), mask_p2);
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * get_adjusted_faction_color(_linear(vec3_color_2.rgb)), mask_p3);
  	}
  	else
  	{
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_0.rgb), mask_p1);
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_1.rgb), mask_p2);
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_2.rgb), mask_p3);
  	}

  	vec3 N = getTSNormal(inputs.tex_coord.xy);

  	vec3 pixel_normal = normalize(basis * normalize(N));

    StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

    vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

    vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

    return vec4(ldr_linear_col, alpha);
  } else if (i_technique == 9) // Full Tint
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

  	vec3 light_vector = normalize(light_position0.xyz - inputs.position);

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);

    float alpha;

    if (b_use_opacity)
    {
      alpha = check_alpha(texture(s_opacity, inputs.tex_coord.xy).r);
    } else
    {
      alpha = check_alpha(diffuse_color.a);
    }

    vec4 specular_color = texture(s_specular_color, inputs.tex_coord.xy);

    float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

    smoothness = _linear(smoothness);

    float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

    reflectivity = _linear(reflectivity);

  	float mask_p1 = texture(s_mask1, inputs.tex_coord.xy).r;
  	float mask_p2 = texture(s_mask2, inputs.tex_coord.xy).r;
  	float mask_p3 = texture(s_mask3, inputs.tex_coord.xy).r;

  	if (b_faction_coloring)
  	{
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_0.rgb), mask_p1);
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_1.rgb), mask_p2);
  		diffuse_color.rgb = mix(diffuse_color.rgb, diffuse_color.rgb * _linear(vec3_color_2.rgb), mask_p3);
  	}

  	vec3 N = getTSNormal(inputs.tex_coord.xy);
  	vec3 pixel_normal = normalize(basis * normalize(N));

    StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

    vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

    vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

    return vec4(ldr_linear_col, alpha);
  } else if (i_technique == 10) // Glossiness
  {
    vec4 glossiness_p = texture(s_smoothness, inputs.tex_coord.xy);

    return vec4(_linear(glossiness_p.rrr), 1.0);
  } else if (i_technique == 11) // Mask 1
  {
    vec4 faction_p = texture(s_mask1, inputs.tex_coord.xy);

    return vec4(_linear(faction_p.rrr), 1.0);
  } else if (i_technique == 12) // Mask 2
  {
    vec4 faction_p = texture(s_mask2, inputs.tex_coord.xy);

    return vec4(_linear(faction_p.rrr), 1.0);
  } else if (i_technique == 13) // Mask 3
  {
    vec4 faction_p = texture(s_mask3, inputs.tex_coord.xy);

    return vec4(_linear(faction_p.rrr), 1.0);
  } else if (i_technique == 14) // N Dot L
  {
    if (light_absolute0 == false)
    {
      light_position0 = vMatrixI * vec4(light_relative0_x, light_relative0_y, light_relative0_z, 1.0);
    } else {
      light_position0 = vec4(light_absolute0_x, light_absolute0_y, light_absolute0_z, 1.0);
    }

    vec4 diffuse_color = texture(s_diffuse_color, inputs.tex_coord.xy);
    vec3 light_vector = normalize(light_position0.xyz -  inputs.position);

    vec3 N = getTSNormal(inputs.tex_coord.xy);
    vec3 pixel_normal = normalize(basis * normalize(N));

    vec3 ndotl = vec3(clamp(dot(pixel_normal, light_vector), 0.0, 1.0));

    return vec4(ndotl, diffuse_color.a);
  } else if (i_technique == 15) // Normal
  {
    vec3 N = normalSwizzle_UPDATED(getTSNormal(inputs.tex_coord.xy));
    vec3 nN = (N + vec3(1.0, 1.0, 0.0)) / vec3(2.0);
    nN.g = 1.0 - nN.g;
    vec3 normal = vec3(_linear(nN.r), _linear(nN.g), _gamma(nN.b));

    return vec4(normal, 1.0);
  } else if (i_technique == 16) // Reflection
  {
    vec3 pI = normalize(vMatrixI[3].xyz - inputs.position);

    vec3 N = getTSNormal(inputs.tex_coord.xy);

    vec3 nN = normalize(basis * N);
    vec3 refl = -reflect(pI, nN);
    vec3 env = get_environment_color_UPDATED(rotate(refl, f_environment_rotation), 0.0);

    return vec4(env.rgb, 1.0);
  } else if (i_technique == 17) // Reflectivity
  {
    vec4 reflectivity_p = texture(s_reflectivity, inputs.tex_coord.xy);

    return vec4(_linear(reflectivity_p.rrr), 1.0);
  } else if (i_technique == 18) // Solid Alpha
  {
    if (b_use_opacity)
    {
      vec4 Ct = texture(s_opacity, inputs.tex_coord.xy);

      return vec4(_linear(Ct.rrr), 1.0);
    } else
    {
      vec4 Ct = texture(s_diffuse_color, inputs.tex_coord.xy);

      return vec4(_linear(Ct.aaa), 1.0);
    }
  } else if (i_technique == 19) // Specular
  {
    vec4 specular_p = texture(s_specular_color, inputs.tex_coord.xy );

    specular_p.rgb = specular_p.rgb;

    return vec4(specular_p.rgb, 1.0);
  } else if (i_technique == 20) // World Space Normal
  {
    vec3 N = getTSNormal(inputs.tex_coord.xy);

  	vec3 nN = ((normalize(basis * N)) * 0.5) + 0.5;

    return vec4(_linear(nN.rgb), 1.0);
  }
}

void shadeShadow(V2F inputs)
{
}
