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

	mat3 basis = MAXTBN;
	vec3 Np = (texture2D(s_normal_map, iFS_UV.xy)).rgb;
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);

	ps_common_blend_decal(diffuse_color, N, specular_color.rgb, reflectivity, diffuse_color, N, specular_color.rgb, reflectivity, iFS_UV.xy, 0, vec4_uv_rect, 1.0);

	vec3 pixel_normal = normalize(basis * normalize(N));

  StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_color.rgb, specular_color.rgb, pixel_normal, smoothness, reflectivity, vec4(iFS_PointWS.xyz, 1.0));

  vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  gl_FragColor = vec4(ldr_linear_col, alpha);
}
