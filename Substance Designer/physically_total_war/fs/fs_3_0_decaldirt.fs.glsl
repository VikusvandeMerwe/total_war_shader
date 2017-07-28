//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec3 eye_vector = -normalize(vMatrixI[3].xyz - iFS_Wpos);

	vec3 light_vector = normalize(light_position0.xyz - iFS_Wpos);

  vec4 diffuse_colour = texture2D(s_diffuse_colour, iFS_TexCoord.xy);

	alpha_test(diffuse_colour.a);

  diffuse_colour.rgb = _linear(diffuse_colour.rgb);

  vec4 specular_colour = texture2D(s_specular_colour, iFS_TexCoord.xy);

  specular_colour.rgb = _linear(specular_colour.rgb);

  float smoothness = texture2D(s_smoothness, iFS_TexCoord.xy).x;

  smoothness = _linear(smoothness);

  float reflectivity = texture2D(s_reflectivity, iFS_TexCoord.xy).x;

  reflectivity = _linear(reflectivity);

	float mask_p1 = texture2D(s_mask1, iFS_TexCoord.xy).r;
	float mask_p2 = texture2D(s_mask2, iFS_TexCoord.xy).r;
	float	mask_p3 = texture2D(s_mask3, iFS_TexCoord.xy).r;

	diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * _linear(vec4_colour_0.rgb), mask_p1);
	diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * _linear(vec4_colour_1.rgb), mask_p2);
	diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * _linear(vec4_colour_2.rgb), mask_p3);

	mat3 basis = MAXTBN;
	vec3 Np = (texture2D(s_normal_map, iFS_TexCoord.xy)).rgb;
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);

	if (b_do_decal)
	{
		ps_common_blend_decal(diffuse_colour, N, specular_colour.rgb, reflectivity, diffuse_colour, N, specular_colour.rgb, reflectivity, iFS_TexCoord.xy, 0, vec4_uv_rect, 1.0);
	}

	if (b_do_dirt)
	{
		ps_common_blend_dirtmap(diffuse_colour, N, specular_colour.rgb, reflectivity, diffuse_colour, N, specular_colour.rgb, reflectivity, iFS_TexCoord.xy, vec2(f_uv_offset_u, f_uv_offset_v));
	}

	vec3 pixel_normal = normalize(basis * normalize(N));

  StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_colour.rgb, specular_colour.rgb, pixel_normal, smoothness, reflectivity, vec4(iFS_Wpos.xyz, 1.0));

  vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  gl_FragColor = vec4(_gamma(ldr_linear_col), 1.0);
}
