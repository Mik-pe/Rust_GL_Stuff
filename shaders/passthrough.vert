#version 330

void main() 
{
	vec2 outUV = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
	gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
}