#version 450
layout(location = 0) in vec3 a_pos;

out gl_PerVertex {
	vec4 gl_Position;
};


void main() 
{
	vec3 pos = a_pos;
	pos = vec3(0.5) - pos;
	gl_Position = vec4(pos, 1.0);
}