export const glLambertVS = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
} model;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

out vec3 vWorldNorm;
out vec3 vVertColor;

void main() {
  vec4 worldPos = model.world * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
  vWorldNorm = (model.world * vec4(normal, 0.0)).xyz;
  vVertColor = color;
}
`

export const glLambertFS = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
} lighting;

in vec3 vWorldNorm;
in vec3 vVertColor;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vWorldNorm);
  vec3 L = normalize(-lighting.direction.xyz);
  float NdotL = max(dot(N, L), 0.0);
  vec3 diffuse = lighting.dirColor.rgb * NdotL;
  vec3 ambient = lighting.ambientColor.rgb;
  vec3 finalColor = model.color.rgb * vVertColor * (diffuse + ambient);
  fragColor = vec4(finalColor, 1.0);
}
`

export const glSkinnedLambertVS = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
} model;

layout(std140) uniform JointMatrices {
  mat4 jointMatrices[128];
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in uvec4 joints;
layout(location = 4) in vec4 weights;

out vec3 vWorldNorm;
out vec3 vVertColor;

void main() {
  mat4 skinMat =
    weights.x * jointMatrices[joints.x] +
    weights.y * jointMatrices[joints.y] +
    weights.z * jointMatrices[joints.z] +
    weights.w * jointMatrices[joints.w];

  vec4 worldPos = model.world * skinMat * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
  vWorldNorm = (model.world * skinMat * vec4(normal, 0.0)).xyz;
  vVertColor = color;
}
`
