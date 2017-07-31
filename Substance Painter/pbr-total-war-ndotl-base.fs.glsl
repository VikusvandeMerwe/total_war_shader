// Shader entry point
vec4 shade(V2F inputs)
{
  vec4 diffuse_colour = texture(s_diffuse_colour, inputs.tex_coord.xy);
  vec3 light_vector = normalize(light_position0.xyz -  inputs.position);

  mat3 MAXTBN = mat3(normalize(inputs.tangent), normalize(inputs.normal), normalize(inputs.bitangent));

  mat3 basis = MAXTBN;
  vec4 Np = texture(s_normal_map, inputs.tex_coord.xy);
  vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);
  vec3 pixel_normal = normalize(basis * normalize(N));

  vec3 ndotl = vec3(clamp(dot(pixel_normal, light_vector), 0.0, 1.0));

  return vec4(_gamma(ndotl), diffuse_colour.a);
}
