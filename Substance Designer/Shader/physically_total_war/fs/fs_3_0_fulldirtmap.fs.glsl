//////////////////////////////// Fragment shader
#version 120
#extension GL_ARB_shader_texture_lod : require

#include "computation.fs.glsl"

void main()
{
  if (f_version != internal_version)
  {
    discard;
  }

  vec3 eye_vector = -normalize(viewInverseMatrix[3].xyz - iFS_PointWS);

	vec3 light_vector = normalize(light_position0.xyz - iFS_PointWS);

  vec4 diffuse_color = texture2D(s_diffuse_color, iFS_UV.xy);

	float alpha = check_alpha(diffuse_color.a);

  diffuse_color.rgb = _linear(diffuse_color.rgb);

  vec4 specular_color = texture2D(s_specular_color, iFS_UV.xy);

  specular_color.rgb = _linear(specular_color.rgb);

  float smoothness = texture2D(s_smoothness, iFS_UV.xy).x;

  smoothness = _linear(smoothness);

  float reflectivity = texture2D(s_reflectivity, iFS_UV.xy).x;

  reflectivity = _linear(reflectivity);

	vec4 dirtmap = texture2D(s_dirtmap_uv2, iFS_UV.xy * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));
  float alpha_mask = texture2D(s_alpha_mask, iFS_UV.xy).a;

  float blend_amount = alpha_mask;

  float hardness = 1.0;

  float blend_2 = blend_amount * mix(1.0, dirtmap.a, blend_amount);

  blend_amount = clamp(((blend_2 - 0.5) * hardness) + 0.5, 0.0, 1.0);

  diffuse_color.rgb = diffuse_color.rgb * (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));
	specular_color.rgb *= (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));


	mat3 basis = MAXTBN;
	vec4 Np = (texture2D(s_normal_map, iFS_UV.xy));
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);
	vec3 pixel_normal = normalize(basis * normalize(N));

  StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(iFS_PointWS.xyz, 1.0));

  vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  gl_FragColor = vec4(ldr_linear_col, alpha);
}
