// Shader entry point
vec4 shade(V2F inputs)
{
  vec4 specular_p = texture(s_specular_colour, inputs.tex_coord.xy );

  return vec4(specular_p.rgb, 1.0);
}
