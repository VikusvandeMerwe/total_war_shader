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

	alpha_test ( diffuse_colour.a );

  diffuse_colour.rgb = _linear(diffuse_colour.rgb);

  vec4 specular_colour = texture2D(s_specular_colour, iFS_TexCoord.xy);

  specular_colour.rgb = _linear(specular_colour.rgb);

  float smoothness = texture2D(s_smoothness, iFS_TexCoord.xy).x;

  smoothness = _linear(smoothness);

  float reflectivity = texture2D(s_reflectivity, iFS_TexCoord.xy).x;

  reflectivity = _linear(reflectivity);

	mat3 basis = MAXTBN;
	vec3 Np = (texture2D(s_normal_map, iFS_TexCoord.xy)).rgb;
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);

	ps_common_blend_decal(diffuse_colour, N, specular_colour.rgb, reflectivity, diffuse_colour, N, specular_colour.rgb, reflectivity, iFS_TexCoord.xy, 0, vec4_uv_rect, 1 - iFS_Color.a);


	vec3 pixel_normal = normalize(basis * normalize(N));

  StandardLightingModelMaterial_UPDATED standard_mat = create_standard_lighting_material_UPDATED(diffuse_colour.rgb, specular_colour.rgb, pixel_normal, smoothness, reflectivity, vec4(iFS_Wpos.xyz, 1.0));

  vec3 hdr_linear_col = standard_lighting_model_directional_light_UPDATED(light_color0, light_vector, eye_vector, standard_mat);

  vec3 ldr_linear_col = clamp(tone_map_linear_hdr_pixel_value(hdr_linear_col), 0.0, 1.0);

  gl_FragColor = vec4(_gamma(ldr_linear_col), 1.0);
}
