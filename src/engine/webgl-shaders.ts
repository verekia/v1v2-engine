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

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP;
  vec4 shadowParams;
} lighting;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

out vec3 vWorldNorm;
out vec3 vVertColor;
out vec3 vShadowCoord;

void main() {
  vec4 worldPos = model.world * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
  vec3 worldNorm = (model.world * vec4(normal, 0.0)).xyz;
  vWorldNorm = worldNorm;
  vVertColor = color;
  // Shadow coord: offset along normal to reduce light bleeding at contact edges
  float normalBias = lighting.shadowParams.y;
  vec3 shadowPos = worldPos.xyz + normalize(worldNorm) * normalBias;
  vec4 lightClip = lighting.lightVP * vec4(shadowPos, 1.0);
  vShadowCoord = vec3(
    lightClip.x * 0.5 + 0.5,
    lightClip.y * 0.5 + 0.5,
    lightClip.z * 0.5 + 0.5
  );
}
`

export const glLambertFS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2DShadow;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP;
  vec4 shadowParams;
} lighting;

uniform sampler2DShadow uShadowMap;

in vec3 vWorldNorm;
in vec3 vVertColor;
in vec3 vShadowCoord;

out vec4 fragColor;

// Poisson disk samples (9 taps)
const vec2 POISSON_DISK[9] = vec2[9](
  vec2(-0.7071,  0.7071),
  vec2( 0.0,    -0.8750),
  vec2( 0.5303,  0.5303),
  vec2(-0.6250,  0.0),
  vec2( 0.8660, -0.25),
  vec2(-0.25,   -0.4330),
  vec2( 0.3536,  0.3536),
  vec2(-0.4330,  0.25),
  vec2( 0.125,  -0.2165)
);

float pcfShadow(vec3 coord) {
  float bias = lighting.shadowParams.x;
  float texelSize = lighting.shadowParams.z;
  float enabled = lighting.shadowParams.w;
  if (enabled < 0.5) return 1.0;
  if (coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0 || coord.z < 0.0 || coord.z > 1.0) return 1.0;
  float refDepth = coord.z - bias;
  float rnd = fract(sin(dot(coord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float angle = rnd * 6.2831853;
  float cosA = cos(angle);
  float sinA = sin(angle);
  mat2 rotMat = mat2(cosA, sinA, -sinA, cosA);
  float shadow = 0.0;
  float spread = texelSize * 1.0;
  for (int i = 0; i < 9; i++) {
    vec2 offset = rotMat * POISSON_DISK[i] * spread;
    shadow += texture(uShadowMap, vec3(coord.xy + offset, refDepth));
  }
  return shadow / 9.0;
}

void main() {
  vec3 N = normalize(vWorldNorm);
  vec3 L = normalize(-lighting.direction.xyz);
  float NdotL = max(dot(N, L), 0.0);
  float shadow = pcfShadow(vShadowCoord);
  vec3 diffuse = lighting.dirColor.rgb * NdotL * shadow;
  vec3 ambient = lighting.ambientColor.rgb;
  vec3 finalColor = model.color.rgb * vVertColor * (diffuse + ambient);
  fragColor = vec4(finalColor, model.color.a);
}
`

export const glUnlitVS = /* glsl */ `#version 300 es
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

out vec3 vVertColor;

void main() {
  vec4 worldPos = model.world * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
  vVertColor = color;
}
`

export const glUnlitFS = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
} model;

in vec3 vVertColor;

out vec4 fragColor;

void main() {
  fragColor = vec4(model.color.rgb * vVertColor, model.color.a);
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

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP;
  vec4 shadowParams;
} lighting;

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
out vec3 vShadowCoord;

void main() {
  mat4 skinMat =
    weights.x * jointMatrices[joints.x] +
    weights.y * jointMatrices[joints.y] +
    weights.z * jointMatrices[joints.z] +
    weights.w * jointMatrices[joints.w];

  vec4 worldPos = model.world * skinMat * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
  vec3 worldNorm = (model.world * skinMat * vec4(normal, 0.0)).xyz;
  vWorldNorm = worldNorm;
  vVertColor = color;
  float normalBias = lighting.shadowParams.y;
  vec3 shadowPos = worldPos.xyz + normalize(worldNorm) * normalBias;
  vec4 lightClip = lighting.lightVP * vec4(shadowPos, 1.0);
  vShadowCoord = vec3(
    lightClip.x * 0.5 + 0.5,
    lightClip.y * 0.5 + 0.5,
    lightClip.z * 0.5 + 0.5
  );
}
`

// ── Shadow depth shaders ──────────────────────────────────────────────

export const glShadowDepthVS = /* glsl */ `#version 300 es
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

void main() {
  vec4 worldPos = model.world * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
}
`

export const glSkinnedShadowDepthVS = /* glsl */ `#version 300 es
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
layout(location = 3) in uvec4 joints;
layout(location = 4) in vec4 weights;

void main() {
  mat4 skinMat =
    weights.x * jointMatrices[joints.x] +
    weights.y * jointMatrices[joints.y] +
    weights.z * jointMatrices[joints.z] +
    weights.w * jointMatrices[joints.w];

  vec4 worldPos = model.world * skinMat * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
}
`

export const glShadowDepthFS = /* glsl */ `#version 300 es
precision highp float;

out vec4 fragColor;

void main() {
  fragColor = vec4(1.0);
}
`
