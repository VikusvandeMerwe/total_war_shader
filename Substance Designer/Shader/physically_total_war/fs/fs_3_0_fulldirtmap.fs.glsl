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

  vec3 eye_vector = -normalize(vMatrixI[3].xyz - iFS_Wpos);

	vec3 light_vector = normalize(light_position0.xyz - iFS_Wpos);

  vec4 diffuse_colour = texture2D(s_diffuse_colour, iFS_TexCoord.xy);

	float alpha = check_alpha(diffuse_colour.a);

  diffuse_colour.rgb = _linear(diffuse_colour.rgb);

  vec4 specular_colour = texture2D(s_specular_colour, iFS_TexCoord.xy);

  specular_colour.rgb = _linear(specular_colour.rgb);

  float smoothness = texture2D(s_smoothness, iFS_TexCoord.xy).x;

  smoothness = _linear(smoothness);

  float reflectivity = texture2D(s_reflectivity, iFS_TexCoord.xy).x;

  reflectivity = _linear(reflectivity);

	vec4 dirtmap = texture2D(s_dirtmap_uv2, iFS_TexCoord.xy * vec2(f_uv2_tile_interval_u, f_uv2_tile_interval_v));
  float alpha_mask = texture2D(s_alpha_mask, iFS_TexCoord.xy).a;

  float blend_amount = alpha_mask;

  float hardness = 1.0;

  float blend_2 = blend_amount * mix(1.0, dirtmap.a, blend_amount);

  blend_amount = clamp(((blend_2 - 0.5) * hardness) + 0.5, 0.0, 1.0);

  diffuse_colour.rgb = diffuse_colour.rgb * (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));
	specular_colour.rgb *= (mix(dirtmap.rgb, vec3(1.0, 1.0, 1.0), blend_amount));


	mat3 basis = MAXTBN;
	vec4 Np = (texture2D(s_normal_map, iFS_TexCoord.xy));
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);
	vec3 pixel_normal = normalize(basis * normalize(N));

  StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_colour.rgb, specular_colour.rgb, pixel_normal, smoothness, reflectivity, vec4(iFS_Wpos.xyz, 1.0));

  vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  gl_FragColor = vec4(_gamma(ldr_linear_col), alpha);
}
