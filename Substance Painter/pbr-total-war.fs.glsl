#version 130

//- Tobias Czyzewski Total War Shader
//- =================================

struct V2F {
  vec3 normal;               // interpolated normal
  vec3 tangent;              // interpolated tangent
  vec3 bitangent;            // interpolated bitangent
  vec3 position;             // interpolated position
  vec4 color[1];             // interpolated vertex colors (color0)
  vec2 tex_coord;            // interpolated texture coordinates (uv0)
  vec2 multi_tex_coord[8];   // interpolated texture coordinates (uv0-uv7)
};

//- Shader entry point
vec4 shade(V2F inputs)
{
  return vec4(1.0, 0.0, 1.0, 1.0);
}
