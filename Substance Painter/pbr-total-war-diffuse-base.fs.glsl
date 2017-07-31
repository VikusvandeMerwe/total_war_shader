// Shader entry point
vec4 shade(V2F inputs)
{
  vec4 diffuse_colour = texture(s_diffuse_colour, inputs.tex_coord.xy);

	alpha_test(diffuse_colour.a);

  return vec4(diffuse_colour.rgb, 1.0);
}
