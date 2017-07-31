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
	diffuse_colour.rgb *= ao.rgb;
	specular_colour.rgb *= ao.rgb;

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
