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

	vec3 ao = texture(s_ambient_occlusion, inputs.tex_coord.xy).rgb;
	float mask_p1 = texture(s_mask1, inputs.tex_coord.xy).r;
	float mask_p2 = texture(s_mask2, inputs.tex_coord.xy).r;
	float mask_p3 = texture(s_mask3, inputs.tex_coord.xy).r;

  mat3 MAXTBN = mat3(normalize(inputs.tangent), normalize(inputs.normal), normalize(inputs.bitangent));

	mat3 basis = MAXTBN;
	vec4 Np = (texture(s_normal_map, inputs.tex_coord.xy));
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);

	if (b_do_decal)
	{
		ps_common_blend_decal(diffuse_colour, N, specular_colour.rgb, reflectivity, diffuse_colour, N, specular_colour.rgb, reflectivity, inputs.tex_coord.xy, 0.0, vec4_uv_rect, 1.0);
	}

	if (b_do_dirt)
	{
		ps_common_blend_dirtmap(diffuse_colour, N, specular_colour.rgb, reflectivity, diffuse_colour, N, specular_colour.rgb, reflectivity, inputs.tex_coord.xy, vec2(f_uv_offset_u, f_uv_offset_v));
	}

	vec3 pixel_normal = normalize(basis * normalize(N));

  SkinLightingModelMaterial skin_mat = create_skin_lighting_material(vec2(smoothness, reflectivity), vec3(mask_p1, mask_p2, mask_p3), diffuse_colour.rgb, specular_colour.rgb, pixel_normal, vec4(inputs.position.xyz, 1.0));

  vec3 hdr_linear_col = skin_lighting_model_directional_light(light_color0, light_vector, eye_vector, skin_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  return vec4(ldr_linear_col, 1.0);
}
