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

	float mask_p1 = texture(s_mask1, inputs.tex_coord.xy).r;
	float mask_p2 = texture(s_mask2, inputs.tex_coord.xy).r;
	float mask_p3 = texture(s_mask3, inputs.tex_coord.xy).r;


	if (b_faction_colouring)
	{
		diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * get_adjusted_faction_colour(_linear(vec4_colour_0.rgb)), mask_p1);
		diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * get_adjusted_faction_colour(_linear(vec4_colour_1.rgb)), mask_p2);
		diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * get_adjusted_faction_colour(_linear(vec4_colour_2.rgb)), mask_p3);
	}
	else
	{
		diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * _linear(vec4_colour_0.rgb), mask_p1);
		diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * _linear(vec4_colour_1.rgb), mask_p2);
		diffuse_colour.rgb = mix(diffuse_colour.rgb, diffuse_colour.rgb * _linear(vec4_colour_2.rgb), mask_p3);
	}

  mat3 MAXTBN = mat3(normalize(inputs.tangent), normalize(inputs.normal), normalize(inputs.bitangent));

	mat3 basis = MAXTBN;
	vec3 Np = (texture(s_normal_map, inputs.tex_coord.xy)).rgb;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);

	vec3 pixel_normal = normalize(basis * normalize(N));

  StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_colour.rgb, specular_colour.rgb, pixel_normal, smoothness, reflectivity, vec4(inputs.position.xyz, 1.0));

  vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  return vec4(ldr_linear_col, 1.0);
}
