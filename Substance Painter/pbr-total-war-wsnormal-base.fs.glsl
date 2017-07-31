// Shader entry point
vec4 shade(V2F inputs)
{
  mat3 MAXTBN = mat3(normalize(inputs.tangent), normalize(inputs.normal), normalize(inputs.bitangent));

  mat3 basis = MAXTBN;
  vec3 Np = (texture(s_normal_map, inputs.tex_coord.xy)).rgb;
	vec3 N = normalSwizzle((Np.rgb * 2.0) - 1.0);

  if (true)
  {
    vec3 N2p = (texture(s_d_normal, inputs.tex_coord.xy * 1.0)).rgb;
    vec3 N2 = normalSwizzle((N2p.rgb * 2.0) - 1.0);
    N = vec3(N.x + (N2.x * 1.0), N.y + (N2.y * 1.0), N.z);
  }

	vec3 nN = ((normalize(basis * N)) * 0.5) + 0.5;

  return vec4(nN.rgb, 1.0);
}
