// Tobias Czyzewski Total War Shader
// =================================

// #version 330

// Vertex shader output
struct V2F {
  vec3 normal;               // interpolated normal
  vec3 tangent;              // interpolated tangent
  vec3 bitangent;            // interpolated bitangent
  vec3 position;             // interpolated position
  vec4 color[1];             // interpolated vertex colors (color0)
  vec2 tex_coord;            // interpolated texture coordinates (uv0)
  vec2 multi_tex_coord[8];   // interpolated texture coordinates (uv0-uv7)
};

//: param auto camera_view_matrix_it
uniform mat4 camera_view_matrix_it;

mat4 vMatrixI = transpose(camera_view_matrix_it);

//: param custom { "default": 0, "label": "Light Position X", "min": -100, "max": 100 }
uniform int light_position0_x;
//: param custom { "default": 0, "label": "Light Position Y", "min": -100, "max": 100 }
uniform int light_position0_y;
//: param custom { "default": 0, "label": "Light Position Z", "min": -100, "max": 100 }
uniform int light_position0_z;
//: param custom { "default": [1.0, 1.0, 1.0], "label": "Light Color", "widget": "color" }
uniform vec3 light_color0;

//: param auto channel_diffuse
uniform sampler2D s_diffuse_colour;
//: param auto channel_glossiness
uniform sampler2D s_smoothness;
//: param auto channel_normal
uniform sampler2D s_normal_map;
//: param auto channel_specularlevel
uniform sampler2D s_reflectivity;
//: param auto channel_specular
uniform sampler2D s_specular_colour;
//: param auto channel_user0
uniform sampler2D s_alpha_mask;
//: param auto channel_user1
uniform sampler2D s_mask1;
//: param auto channel_user2
uniform sampler2D s_mask2;
//: param auto channel_user3
uniform sampler2D s_mask3;
//: param auto channel_user4
uniform sampler2D s_decal_diffuse;
//: param auto channel_user5
uniform sampler2D s_decal_normal;
//: param auto channel_user6
uniform sampler2D s_decal_mask;
//: param auto channel_user7
uniform sampler2D s_decal_dirtmap;
//: param auto channel_anisotropyangle
uniform sampler2D s_decal_dirtmask;
//: param auto channel_ambientocclusion
uniform sampler2D s_ambient_occlusion;
//: param auto channel_transmissive
uniform sampler2D s_ambient;
//: param auto channel_reflection
uniform sampler2D s_environment_map;
//: param auto channel_blendingmask
uniform sampler2D s_dirtmap_uv2;

//: param custom { "default": true, "label": "Shadows" }
uniform bool b_shadows;
//: param custom { "default": true, "label": "Alpha Off" }
uniform bool b_alpha_off;
//: param custom { "default": 0, "label": "Alpha Mode", "min": 0, "max": 2 }
uniform int i_alpha_mode;
//: param custom { "default": true, "label": "Faction Colouring" }
uniform bool b_faction_colouring;
//: param custom { "default": 0.0, "label": "UV Offset U", "min": -500.0, "max": 500.0, "step": 1.0 }
uniform float f_uv_offset_u;
//: param custom { "default": 0.0, "label": "UV Offset V", "min": -500.0, "max": 500.0, "step": 1.0 }
uniform float f_uv_offset_v;
//: param custom { "default": false, "label": "Enable Decal" }
uniform bool b_do_decal;
//: param custom { "default": false, "label": "Enable Dirt" }
uniform bool b_do_dirt;
//: param custom { "default": [0.5, 0.1, 0.1, 1.0], "label": "Tint Color 1", "widget": "color" }
uniform vec4 vec4_colour_0;
//: param custom { "default": [0.3, 0.6, 0.5, 1.0], "label": "Tint Color 2", "widget": "color" }
uniform vec4 vec4_colour_1;
//: param custom { "default": [0.5, 0.2, 0.1, 1.0], "label": "Tint Color 3", "widget": "color" }
uniform vec4 vec4_colour_2;
//: param custom { "default": 1, "label": "Dirt Tile U", "min": 0, "max": 1 }
uniform int i_random_tile_u;
//: param custom { "default": 1, "label": "Dirt Tile V", "min": 0, "max": 1 }
uniform int i_random_tile_v;
//: param custom { "default": 4.0, "label": "Detail Tile U", "min": -100.0, "max": 100.0, "step": 1.0 }
uniform float f_uv2_tile_interval_u;
//: param custom { "default": 4.0, "label": "Detail Tile V", "min": -100.0, "max": 100.0, "step": 1.0 }
uniform float f_uv2_tile_interval_v;

/////////////////////////
// Parameters
/////////////////////////

vec4 light_position0 = vMatrixI * vec4(light_position0_x, light_position0_y, light_position0_z, 1.0);

const float pi = 3.14159265;
const float one_over_pi = 1.0 / 3.14159265;
const float real_approx_zero = 0.001;
const float texture_alpha_ref = 0.5;

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

float get_luminance(in vec3 colour)
{
	vec3 lumCoeff = vec3(0.299, 0.587, 0.114);
	float luminance = dot(colour, lumCoeff);
	return clamp(luminance, 0.0, 1.0);
}

vec3 get_adjusted_faction_colour(in vec3 colour)
{
	vec3 fc = colour;
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
// Forward Declarations
/////////////////////////

vec3 tone_map_linear_hdr_pixel_value(in vec3 linear_hdr_pixel_val);
vec4 HDR_RGB_To_HDR_CIE_Log_Y_xy(in vec3 linear_colour_val);
vec4 tone_map_HDR_CIE_Log_Y_xy_To_LDR_CIE_Yxy(in vec4 hdr_LogYxy);
vec4 LDR_CIE_Yxy_To_Linear_LDR_RGB(in vec4 ldr_cie_Yxy);
float get_scurve_y_pos(const float x_coord);

/////////////////////////
// Lighting Functions
/////////////////////////

vec3 get_environment_colour_UPDATED(in vec3 direction , in float lod)
{
	// return _linear(textureLod(s_environment_map, texcoordEnvSwizzle(direction).xy, lod).rgb);
  return vec3(0.0, 0.0, 0.0);
}

vec3 cube_ambient(in vec3 N)
{
  // return _linear(texture(s_ambient, texcoordEnvSwizzle(N).xy).rgb);
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

void ps_common_blend_decal(in vec4 colour, in vec3 normal, in vec3 specular, in float reflectivity, out vec4 ocolour, out vec3 onormal, out vec3 ospecular, out float oreflectivity, in vec2 uv, in float decal_index, in vec4 uv_rect_coords, in float valpha)
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
	float decal_mask = texture(s_decal_mask, uv).a;

	float decalblend = decal_mask * decal_diffuse.a * valpha;
	oreflectivity = mix(reflectivity, reflectivity * 0.5, decalblend);
	ocolour = vec4(1.0);
	onormal = vec3(0.0, 0.0, 1.0);
	ospecular = mix(specular, decal_diffuse.rgb, decalblend);

	ocolour.rgb = mix(colour.rgb, decal_diffuse.rgb, decalblend);

	onormal.xyz = mix(onormal.xyz, decal_normal.rgb, decalblend);
	onormal.xyz = vec3(normal.xy + onormal.xy, normal.z);
}

void ps_common_blend_dirtmap(inout vec4 colour, inout vec3 normal, in vec3 specular, inout float reflectivity, out vec4 ocolour, out vec3 onormal, out vec3 ospecular, out float oreflectivity, in vec2 uv, in vec2 uv_offset)
{
	uv_offset = uv_offset * vec2(i_random_tile_u, i_random_tile_v);

	float mask_alpha = texture(s_decal_dirtmask, uv).a;
	vec4 dirtmap = texture(s_decal_dirtmap, (uv + uv_offset) * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));

	float d_strength = 1.0;

	// vec2 dirt_normal = (vec2(dirtmap.r, dirtmap.g) * 2.0) - 1.0;
  vec2 dirt_normal = vec2(dirtmap.r, dirtmap.g);

	float dirt_alpha = dirtmap.a;
	float dirt_alpha_blend = mask_alpha * dirt_alpha * d_strength;

	vec3 dirt_colour = vec3(0.03, 0.03, 0.02);
	ocolour = colour;
	onormal = normal;

	ocolour.rgb = mix(colour.rgb, dirt_colour, dirt_alpha_blend);

	ospecular = mix(specular, dirt_colour, dirt_alpha_blend);

	onormal.xz += (dirt_normal.xy * mask_alpha * d_strength);
	onormal = normalize(onormal);

	oreflectivity = reflectivity;
}

void ps_common_blend_vfx(inout vec4 colour, inout vec3 normal, in vec3 specular, inout float reflectivity, out vec4 ocolour, out vec3 onormal, out vec3 ospecular, out float oreflectivity, in vec2 uv, in vec2 uv_offset)
{
	uv_offset = uv_offset * vec2(i_random_tile_u, i_random_tile_v);

	vec4 dirtmap = texture(s_decal_dirtmap, (uv + uv_offset) * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));

	ocolour = vec4(mix(colour.rgb, dirtmap.rgb, dirtmap.a), 1.0);

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
	vec3 specular_colour;
	vec3 Normal;
	float Depth;
	float Shadow;
	float SSAO;
};

SkinLightingModelMaterial create_skin_lighting_material(in vec2 _MaterialMap, in vec3 _SkinMap, in vec3 _Color, in vec3 _specular_colour, in vec3 _Normal, in vec4 _worldposition)
{
	SkinLightingModelMaterial material;

	material.Gloss = _MaterialMap.x;
	material.SpecularLevel = _MaterialMap.y;
	material.RimMask = _SkinMap.x;
	material.SubSurfaceStrength = _SkinMap.y;
	material.BackScatterStrength = _SkinMap.z;
	material.Color = _Color;
	material.specular_colour = _specular_colour;
	material.Normal = normalize(_Normal);
	material.Depth = 1.0;
	material.Shadow = 1.0;
	material.SSAO = 1.0;

	return material;
}

vec3 skin_shading(in vec3 L, in vec3 N, in vec3 V, in float sss_strength, in vec3 colour1, in vec3 colour2)
{
	float ndotl = dot(N, -L);

	vec3 diff1 = vec3(ndotl * clamp(((ndotl * 0.8) + 0.3) / 1.44, 0.0, 1.0));
	vec3 diff2 = colour1 * (clamp(((ndotl * 0.9) + 0.5) / 1.44, 0.0, 1.0)) * clamp(1.0 - (diff1 + 0.3), 0.0, 1.0);
	vec3 diff3 = colour2 * (clamp(((ndotl * 0.3) + 0.3) / 2.25, 0.0, 1.0)) * (1.0 - diff1) * (1.0 - diff2);

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

	vec3 backscatter_colour = mix(LightColor, LightColor * vec3(0.7, 0.0, 0.0), skinlm_material.SubSurfaceStrength) * diffuse_scale_factor * backscatter * skinlm_material.Shadow;

	dlight_diffuse += (skinlm_material.Color.rgb * backscatter_colour * get_skin_dlight_diffuse_scaler());

	vec3  env_light_diffuse = skinlm_material.Color.rgb * cube_ambient(skinlm_material.Normal).rgb;

  float kspec = phong_specular(normalised_view_dir, skinlm_material.Normal, mix(1.0, 128.0, (skinlm_material.Gloss * skinlm_material.Gloss)), normalised_light_dir );

	vec3 dlight_specular = skinlm_material.SpecularLevel * kspec * LightColor * diffuse_scale_factor * get_skin_dlight_specular_scaler();

  vec3 reflected_view_vec = reflect(-normalised_view_dir, skinlm_material.Normal);

 vec3 rim_env_colour = cube_ambient(reflected_view_vec).rgb;

	float rimfresnel = clamp(1.0 - dot(-normalised_view_dir, skinlm_material.Normal), 0.0, 1.0);

  //1.5 is a scalar in order that we can get a brighter rim by setting the texture to be white. Previously the maximum brightness was quite dull.
	float riml = clamp(pow(rimfresnel, 2.0), 0.0, 1.0) * skinlm_material.RimMask * 1.5 * get_skin_dlight_rim_scaler();

	float upness = max(dot(normalize(skinlm_material.Normal + vec3(0.0, 0.75, 0.0)), vec3(0.0, 1.0, 0.0)), 0.0);

  vec3  env_light_specular = (riml * upness * rim_env_colour);

	float   shadow_attenuation      = skinlm_material.Shadow;

  return (skinlm_material.SSAO * (env_light_diffuse + env_light_specular)) + (shadow_attenuation * (dlight_specular + dlight_diffuse));
}

/////////////////////////
// Standard Lightning Model
/////////////////////////

struct StandardLightingModelMaterial_UPDATED
{
	vec3 Diffuse_Colour;
  vec3 Specular_Colour;
	vec3 Normal;
	float Smoothness;
  float Reflectivity;
	float Depth;
	float Shadow;
	float SSAO;
};

StandardLightingModelMaterial_UPDATED create_standard_lighting_material_UPDATED(in vec3 _DiffuseColor, in vec3 _Specular_Colour, in vec3 _Normal, in float _Smoothness, in float _Reflectivity, in vec4 _worldposition)
{
  StandardLightingModelMaterial_UPDATED material;

  material.Diffuse_Colour = _DiffuseColor;
  material.Specular_Colour = _Specular_Colour;
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
	float	towards_diffuse_surface	= sin(roughness * pi * 0.5); //	( 0 - 1 ) output...
	float	facet_visibility = mix(1.0, n_dot_l, towards_diffuse_surface);

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

  vec3 environment_colour = get_environment_colour_UPDATED(reflected_view_vec, env_map_lod);

  vec3 dlight_pixel_reflectivity = get_reflectivity_dir_light(normalised_light_dir, material.Normal, normalised_view_dir, reflected_view_vec, material.Reflectivity, material.Smoothness, roughness);
  vec3 dlight_specular_colour = dlight_pixel_reflectivity * material.Specular_Colour * LightColor;
  // All photons not accounted for by reflectivity are accounted by scattering. From the energy difference between incoming light and emitted light we could calculate the amount of energy turned into heat. This energy would not be enough to make a viewable difference at standard illumination levels.
  vec3 dlight_material_scattering  = 1.0 - max(dlight_pixel_reflectivity, material.Reflectivity);

  vec3  env_light_pixel_reflectivity = max(vec3(material.Reflectivity), get_reflectivity_env_light(reflected_view_vec, material.Normal, normalised_view_dir, material.Reflectivity, material.Smoothness, roughness));
  vec3  env_light_specular_colour = environment_colour * env_light_pixel_reflectivity * material.Specular_Colour;

	vec3  dlight_diffuse = material.Diffuse_Colour * normal_dot_light_vec * LightColor * dlight_material_scattering;

	vec3 ambient_colour = cube_ambient(material.Normal);

  vec3 env_light_diffuse = ambient_colour * material.Diffuse_Colour * (1.0 - material.Reflectivity);

  dlight_diffuse *= diffuse_scale_factor;

	if (!b_shadows)
	{
		material.Shadow = 1.0;
	}
	else
	{
		material.Shadow = material.Shadow;
	}
	float shadow_attenuation = material.Shadow;
	float reflection_shadow_attenuation = (1.0 - ((1.0 - material.Shadow) * 0.75));

  return (material.SSAO * (env_light_diffuse + (reflection_shadow_attenuation * env_light_specular_colour))) + (shadow_attenuation * (dlight_specular_colour + dlight_diffuse));
}

void alpha_test (const float pixel_alpha)
{
	if (b_alpha_off == false && i_alpha_mode == 1)
	{
    if (pixel_alpha - texture_alpha_ref < 0.0)
    {
      discard;
    }
	}
}

/////////////////////////
// Tone Mapper
/////////////////////////

vec4 HDR_RGB_To_HDR_CIE_Log_Y_xy(in vec3 linear_colour_val)
{
	mat3 cie_transform_mat = mat3(0.4124, 0.3576, 0.1805,
										            0.2126, 0.7152, 0.0722,
										            0.0193, 0.1192, 0.9505);

	vec3 cie_XYZ = cie_transform_mat * linear_colour_val;

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
  vec3 eye_vector = -normalize(vMatrixI[3].xyz - inputs.position);

	vec3 light_vector = normalize(light_position0.xyz - inputs.position);

  vec4 diffuse_colour = texture(s_diffuse_colour, inputs.tex_coord.xy);

	alpha_test(diffuse_colour.a);

  vec4 specular_colour = texture(s_specular_colour, inputs.tex_coord.xy);

  float smoothness = texture(s_smoothness, inputs.tex_coord.xy).x;

  smoothness = _linear(smoothness);

  float reflectivity = texture(s_reflectivity, inputs.tex_coord.xy).x;

  reflectivity = _linear(reflectivity);

	vec4 dirtmap = texture(s_dirtmap_uv2, inputs.tex_coord.xy * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));
  float alpha_mask = texture(s_alpha_mask, inputs.tex_coord.xy).a;

  float blend_amount = alpha_mask;

  float hardness = 1.0;

  float blend_2 = blend_amount * mix(1.0, dirtmap.a, blend_amount);

  blend_amount = clamp(((blend_2 - 0.5) * hardness) + 0.5, 0.0, 1.0);

  diffuse_colour.rgb = diffuse_colour.rgb * (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));
	specular_colour.rgb *= (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));

  mat3 MAXTBN = mat3(normalize(inputs.tangent), normalize(inputs.normal), normalize(inputs.bitangent));

	mat3 basis = MAXTBN;
	vec4 Np = (texture(s_normal_map, inputs.tex_coord.xy));
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);
	vec3 pixel_normal = normalize(basis * normalize(N));

  StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_colour.rgb, specular_colour.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

  vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  return vec4(ldr_linear_col, 1.0);
}

void shadeShadow(V2F inputs)
{
}
