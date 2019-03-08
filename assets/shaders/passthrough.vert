#version 450
layout(location = 0) in vec3 a_pos;

out gl_PerVertex {
	vec4 gl_Position;
};


void main() 
{
	// vec2 outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	//gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
	vec3 pos = a_pos;
	pos = vec3(0.5) - pos;
	gl_Position = vec4(pos, 1.0);
}