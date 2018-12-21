#version 330

uniform float time;

out vec4 out_Color;
void main()
{
  float localTime = float(time)/1000.0f;
  out_Color = vec4(0.0f, 1.f, 1.f, 1.0);
}