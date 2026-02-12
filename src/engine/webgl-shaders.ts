export const glLambertVS = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP[4];
  vec4 shadowParams;
  vec4 cascadeSplits;
} lighting;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

out vec3 vWorldNorm;
out vec3 vVertColor;
out vec3 vWorldPos;

void main() {
  vec4 worldPos = model.world * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
  vWorldNorm = (model.world * vec4(normal, 0.0)).xyz;
  vVertColor = color;
  vWorldPos = worldPos.xyz;
}
`

export const glLambertFS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2DShadow;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP[4];
  vec4 shadowParams;
  vec4 cascadeSplits;
} lighting;

uniform sampler2DShadow uShadowMap;

in vec3 vWorldNorm;
in vec3 vVertColor;
in vec3 vWorldPos;

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

const vec2 CASCADE_OFFSET[4] = vec2[4](
  vec2(0.0, 0.0),
  vec2(0.5, 0.0),
  vec2(0.0, 0.5),
  vec2(0.5, 0.5)
);

float csmShadow(vec3 worldPos, vec3 worldNorm) {
  float bias = lighting.shadowParams.x;
  float normalBias = lighting.shadowParams.y;
  float texelSize = lighting.shadowParams.z;
  float enabled = lighting.shadowParams.w;
  if (enabled < 0.5) return 1.0;
  vec4 viewPos = camera.view * vec4(worldPos, 1.0);
  float viewZ = -viewPos.z;
  int cascadeIdx = 0;
  if (viewZ > lighting.cascadeSplits.x) cascadeIdx = 1;
  if (viewZ > lighting.cascadeSplits.y) cascadeIdx = 2;
  if (viewZ > lighting.cascadeSplits.z) cascadeIdx = 3;
  vec3 shadowPos = worldPos + normalize(worldNorm) * normalBias;
  vec4 lightClip = lighting.lightVP[cascadeIdx] * vec4(shadowPos, 1.0);
  vec3 tileCoord = vec3(
    lightClip.x * 0.5 + 0.5,
    lightClip.y * 0.5 + 0.5,
    lightClip.z * 0.5 + 0.5
  );
  if (tileCoord.x < 0.0 || tileCoord.x > 1.0 || tileCoord.y < 0.0 || tileCoord.y > 1.0 || tileCoord.z < 0.0 || tileCoord.z > 1.0) return 1.0;
  vec2 atlasOffset = CASCADE_OFFSET[cascadeIdx];
  vec2 atlasUV = atlasOffset + tileCoord.xy * 0.5;
  float refDepth = tileCoord.z - bias;
  float rnd = fract(sin(dot(atlasUV, vec2(12.9898, 78.233))) * 43758.5453);
  float angle = rnd * 6.2831853;
  float cosA = cos(angle);
  float sinA = sin(angle);
  mat2 rotMat = mat2(cosA, sinA, -sinA, cosA);
  float shadow = 0.0;
  float spread = texelSize * 0.5;
  for (int i = 0; i < 9; i++) {
    vec2 offset = rotMat * POISSON_DISK[i] * spread;
    shadow += texture(uShadowMap, vec3(atlasUV + offset, refDepth));
  }
  return shadow / 9.0;
}

void main() {
  vec3 N = normalize(vWorldNorm);
  vec3 L = normalize(-lighting.direction.xyz);
  float NdotL = max(dot(N, L), 0.0);
  float shadow = csmShadow(vWorldPos, vWorldNorm);
  vec3 diffuse = lighting.dirColor.rgb * NdotL * shadow;
  vec3 ambient = lighting.ambientColor.rgb;
  vec3 finalColor = model.color.rgb * vVertColor * (diffuse + ambient);
  fragColor = vec4(finalColor, model.color.a);
}
`

export const glLambertMRTFS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2DShadow;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP[4];
  vec4 shadowParams;
  vec4 cascadeSplits;
} lighting;

uniform sampler2DShadow uShadowMap;

in vec3 vWorldNorm;
in vec3 vVertColor;
in vec3 vWorldPos;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 bloomColor;
layout(location = 2) out vec4 outlineOut;

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

const vec2 CASCADE_OFFSET[4] = vec2[4](
  vec2(0.0, 0.0),
  vec2(0.5, 0.0),
  vec2(0.0, 0.5),
  vec2(0.5, 0.5)
);

float csmShadow(vec3 worldPos, vec3 worldNorm) {
  float bias = lighting.shadowParams.x;
  float normalBias = lighting.shadowParams.y;
  float texelSize = lighting.shadowParams.z;
  float enabled = lighting.shadowParams.w;
  if (enabled < 0.5) return 1.0;
  vec4 viewPos = camera.view * vec4(worldPos, 1.0);
  float viewZ = -viewPos.z;
  int cascadeIdx = 0;
  if (viewZ > lighting.cascadeSplits.x) cascadeIdx = 1;
  if (viewZ > lighting.cascadeSplits.y) cascadeIdx = 2;
  if (viewZ > lighting.cascadeSplits.z) cascadeIdx = 3;
  vec3 shadowPos = worldPos + normalize(worldNorm) * normalBias;
  vec4 lightClip = lighting.lightVP[cascadeIdx] * vec4(shadowPos, 1.0);
  vec3 tileCoord = vec3(
    lightClip.x * 0.5 + 0.5,
    lightClip.y * 0.5 + 0.5,
    lightClip.z * 0.5 + 0.5
  );
  if (tileCoord.x < 0.0 || tileCoord.x > 1.0 || tileCoord.y < 0.0 || tileCoord.y > 1.0 || tileCoord.z < 0.0 || tileCoord.z > 1.0) return 1.0;
  vec2 atlasOffset = CASCADE_OFFSET[cascadeIdx];
  vec2 atlasUV = atlasOffset + tileCoord.xy * 0.5;
  float refDepth = tileCoord.z - bias;
  float rnd = fract(sin(dot(atlasUV, vec2(12.9898, 78.233))) * 43758.5453);
  float angle = rnd * 6.2831853;
  float cosA = cos(angle);
  float sinA = sin(angle);
  mat2 rotMat = mat2(cosA, sinA, -sinA, cosA);
  float shadow = 0.0;
  float spread = texelSize * 0.5;
  for (int i = 0; i < 9; i++) {
    vec2 offset = rotMat * POISSON_DISK[i] * spread;
    shadow += texture(uShadowMap, vec3(atlasUV + offset, refDepth));
  }
  return shadow / 9.0;
}

void main() {
  vec3 N = normalize(vWorldNorm);
  vec3 L = normalize(-lighting.direction.xyz);
  float NdotL = max(dot(N, L), 0.0);
  float shadow = csmShadow(vWorldPos, vWorldNorm);
  vec3 diffuse = lighting.dirColor.rgb * NdotL * shadow;
  vec3 ambient = lighting.ambientColor.rgb;
  vec3 finalColor = model.color.rgb * vVertColor * (diffuse + ambient);
  vec4 c = vec4(finalColor, model.color.a);
  fragColor = vec4(mix(c.rgb, vec3(1.0), model.bloomWhiten), c.a);
  bloomColor = vec4(c.rgb * model.bloom, c.a);
  outlineOut = vec4(model.outline, model.outlineScale, 0.0, 1.0);
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
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
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
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

in vec3 vVertColor;

out vec4 fragColor;

void main() {
  fragColor = vec4(model.color.rgb * vVertColor, model.color.a);
}
`

export const glUnlitMRTFS = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

in vec3 vVertColor;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 bloomColor;
layout(location = 2) out vec4 outlineOut;

void main() {
  vec4 c = vec4(model.color.rgb * vVertColor, model.color.a);
  fragColor = vec4(mix(c.rgb, vec3(1.0), model.bloomWhiten), c.a);
  bloomColor = vec4(c.rgb * model.bloom, c.a);
  outlineOut = vec4(model.outline, model.outlineScale, 0.0, 1.0);
}
`

export const glTexturedLambertVS = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP[4];
  vec4 shadowParams;
  vec4 cascadeSplits;
} lighting;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;

out vec3 vWorldNorm;
out vec3 vVertColor;
out vec3 vWorldPos;
out vec2 vUV;

void main() {
  vec4 worldPos = model.world * vec4(position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;
  vWorldNorm = (model.world * vec4(normal, 0.0)).xyz;
  vVertColor = color;
  vUV = uv;
  vWorldPos = worldPos.xyz;
}
`

export const glTexturedLambertFS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2DShadow;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP[4];
  vec4 shadowParams;
  vec4 cascadeSplits;
} lighting;

uniform sampler2DShadow uShadowMap;
uniform sampler2D uAoMap;

in vec3 vWorldNorm;
in vec3 vVertColor;
in vec3 vWorldPos;
in vec2 vUV;

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

const vec2 CASCADE_OFFSET[4] = vec2[4](
  vec2(0.0, 0.0),
  vec2(0.5, 0.0),
  vec2(0.0, 0.5),
  vec2(0.5, 0.5)
);

float csmShadow(vec3 worldPos, vec3 worldNorm) {
  float bias = lighting.shadowParams.x;
  float normalBias = lighting.shadowParams.y;
  float texelSize = lighting.shadowParams.z;
  float enabled = lighting.shadowParams.w;
  if (enabled < 0.5) return 1.0;
  vec4 viewPos = camera.view * vec4(worldPos, 1.0);
  float viewZ = -viewPos.z;
  int cascadeIdx = 0;
  if (viewZ > lighting.cascadeSplits.x) cascadeIdx = 1;
  if (viewZ > lighting.cascadeSplits.y) cascadeIdx = 2;
  if (viewZ > lighting.cascadeSplits.z) cascadeIdx = 3;
  vec3 shadowPos = worldPos + normalize(worldNorm) * normalBias;
  vec4 lightClip = lighting.lightVP[cascadeIdx] * vec4(shadowPos, 1.0);
  vec3 tileCoord = vec3(
    lightClip.x * 0.5 + 0.5,
    lightClip.y * 0.5 + 0.5,
    lightClip.z * 0.5 + 0.5
  );
  if (tileCoord.x < 0.0 || tileCoord.x > 1.0 || tileCoord.y < 0.0 || tileCoord.y > 1.0 || tileCoord.z < 0.0 || tileCoord.z > 1.0) return 1.0;
  vec2 atlasOffset = CASCADE_OFFSET[cascadeIdx];
  vec2 atlasUV = atlasOffset + tileCoord.xy * 0.5;
  float refDepth = tileCoord.z - bias;
  float rnd = fract(sin(dot(atlasUV, vec2(12.9898, 78.233))) * 43758.5453);
  float angle = rnd * 6.2831853;
  float cosA = cos(angle);
  float sinA = sin(angle);
  mat2 rotMat = mat2(cosA, sinA, -sinA, cosA);
  float shadow = 0.0;
  float spread = texelSize * 0.5;
  for (int i = 0; i < 9; i++) {
    vec2 offset = rotMat * POISSON_DISK[i] * spread;
    shadow += texture(uShadowMap, vec3(atlasUV + offset, refDepth));
  }
  return shadow / 9.0;
}

void main() {
  vec3 N = normalize(vWorldNorm);
  vec3 L = normalize(-lighting.direction.xyz);
  float NdotL = max(dot(N, L), 0.0);
  float shadow = csmShadow(vWorldPos, vWorldNorm);
  vec3 diffuse = lighting.dirColor.rgb * NdotL * shadow;
  float ao = texture(uAoMap, vUV).r;
  vec3 ambient = lighting.ambientColor.rgb * ao;
  vec3 finalColor = model.color.rgb * vVertColor * (diffuse + ambient);
  fragColor = vec4(finalColor, model.color.a);
}
`

export const glTexturedLambertMRTFS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2DShadow;

layout(std140) uniform Camera {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform Model {
  mat4 world;
  vec4 color;
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP[4];
  vec4 shadowParams;
  vec4 cascadeSplits;
} lighting;

uniform sampler2DShadow uShadowMap;
uniform sampler2D uAoMap;

in vec3 vWorldNorm;
in vec3 vVertColor;
in vec3 vWorldPos;
in vec2 vUV;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 bloomColor;
layout(location = 2) out vec4 outlineOut;

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

const vec2 CASCADE_OFFSET[4] = vec2[4](
  vec2(0.0, 0.0),
  vec2(0.5, 0.0),
  vec2(0.0, 0.5),
  vec2(0.5, 0.5)
);

float csmShadow(vec3 worldPos, vec3 worldNorm) {
  float bias = lighting.shadowParams.x;
  float normalBias = lighting.shadowParams.y;
  float texelSize = lighting.shadowParams.z;
  float enabled = lighting.shadowParams.w;
  if (enabled < 0.5) return 1.0;
  vec4 viewPos = camera.view * vec4(worldPos, 1.0);
  float viewZ = -viewPos.z;
  int cascadeIdx = 0;
  if (viewZ > lighting.cascadeSplits.x) cascadeIdx = 1;
  if (viewZ > lighting.cascadeSplits.y) cascadeIdx = 2;
  if (viewZ > lighting.cascadeSplits.z) cascadeIdx = 3;
  vec3 shadowPos = worldPos + normalize(worldNorm) * normalBias;
  vec4 lightClip = lighting.lightVP[cascadeIdx] * vec4(shadowPos, 1.0);
  vec3 tileCoord = vec3(
    lightClip.x * 0.5 + 0.5,
    lightClip.y * 0.5 + 0.5,
    lightClip.z * 0.5 + 0.5
  );
  if (tileCoord.x < 0.0 || tileCoord.x > 1.0 || tileCoord.y < 0.0 || tileCoord.y > 1.0 || tileCoord.z < 0.0 || tileCoord.z > 1.0) return 1.0;
  vec2 atlasOffset = CASCADE_OFFSET[cascadeIdx];
  vec2 atlasUV = atlasOffset + tileCoord.xy * 0.5;
  float refDepth = tileCoord.z - bias;
  float rnd = fract(sin(dot(atlasUV, vec2(12.9898, 78.233))) * 43758.5453);
  float angle = rnd * 6.2831853;
  float cosA = cos(angle);
  float sinA = sin(angle);
  mat2 rotMat = mat2(cosA, sinA, -sinA, cosA);
  float shadow = 0.0;
  float spread = texelSize * 0.5;
  for (int i = 0; i < 9; i++) {
    vec2 offset = rotMat * POISSON_DISK[i] * spread;
    shadow += texture(uShadowMap, vec3(atlasUV + offset, refDepth));
  }
  return shadow / 9.0;
}

void main() {
  vec3 N = normalize(vWorldNorm);
  vec3 L = normalize(-lighting.direction.xyz);
  float NdotL = max(dot(N, L), 0.0);
  float shadow = csmShadow(vWorldPos, vWorldNorm);
  vec3 diffuse = lighting.dirColor.rgb * NdotL * shadow;
  float ao = texture(uAoMap, vUV).r;
  vec3 ambient = lighting.ambientColor.rgb * ao;
  vec3 finalColor = model.color.rgb * vVertColor * (diffuse + ambient);
  vec4 c = vec4(finalColor, model.color.a);
  fragColor = vec4(mix(c.rgb, vec3(1.0), model.bloomWhiten), c.a);
  bloomColor = vec4(c.rgb * model.bloom, c.a);
  outlineOut = vec4(model.outline, model.outlineScale, 0.0, 1.0);
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
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
} model;

layout(std140) uniform Lighting {
  vec4 direction;
  vec4 dirColor;
  vec4 ambientColor;
  mat4 lightVP[4];
  vec4 shadowParams;
  vec4 cascadeSplits;
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
out vec3 vWorldPos;

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
  vWorldPos = worldPos.xyz;
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
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
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
  float bloom;
  float bloomWhiten;
  float outline;
  float outlineScale;
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

// ── Bloom post-processing shaders ─────────────────────────────────────

export const glBloomDownsampleVS = /* glsl */ `#version 300 es
precision highp float;

void main() {
  float x = float((gl_VertexID << 1) & 2);
  float y = float(gl_VertexID & 2);
  gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
`

export const glBloomDownsampleFS = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D uSrcTex;
uniform vec2 uSrcTexelSize;

out vec4 fragColor;

void main() {
  // pos.xy is in dest pixels; dest = src / 2, so uv = pos.xy * srcTexelSize * 2
  vec2 uv = gl_FragCoord.xy * uSrcTexelSize * 2.0;
  vec2 t = uSrcTexelSize;
  // 4-tap bilinear downsample
  vec4 a = texture(uSrcTex, uv + vec2(-t.x, -t.y) * 0.5);
  vec4 b = texture(uSrcTex, uv + vec2( t.x, -t.y) * 0.5);
  vec4 c = texture(uSrcTex, uv + vec2(-t.x,  t.y) * 0.5);
  vec4 d = texture(uSrcTex, uv + vec2( t.x,  t.y) * 0.5);
  fragColor = (a + b + c + d) * 0.25;
}
`

export const glBloomUpsampleVS = /* glsl */ `#version 300 es
precision highp float;

void main() {
  float x = float((gl_VertexID << 1) & 2);
  float y = float(gl_VertexID & 2);
  gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
`

export const glBloomUpsampleFS = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D uSrcTex;
uniform vec2 uDestTexelSize;
uniform vec2 uOffset;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uDestTexelSize;
  vec2 o = uOffset;
  // 9-tap tent filter
  vec4 color = texture(uSrcTex, uv) * 4.0;
  color += texture(uSrcTex, uv + vec2(-o.x, 0.0)) * 2.0;
  color += texture(uSrcTex, uv + vec2( o.x, 0.0)) * 2.0;
  color += texture(uSrcTex, uv + vec2(0.0, -o.y)) * 2.0;
  color += texture(uSrcTex, uv + vec2(0.0,  o.y)) * 2.0;
  color += texture(uSrcTex, uv + vec2(-o.x, -o.y));
  color += texture(uSrcTex, uv + vec2( o.x, -o.y));
  color += texture(uSrcTex, uv + vec2(-o.x,  o.y));
  color += texture(uSrcTex, uv + vec2( o.x,  o.y));
  fragColor = color / 16.0;
}
`

export const glBloomCompositeVS = /* glsl */ `#version 300 es
precision highp float;

out vec2 vUV;

void main() {
  float x = float((gl_VertexID << 1) & 2);
  float y = float(gl_VertexID & 2);
  gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  vUV = vec2(x, y);
}
`

export const glBloomCompositeFS = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D uSceneTex;
uniform sampler2D uBloomTex;
uniform sampler2D uOutlineTex;
uniform float uIntensity;
uniform float uThreshold;
uniform float uOutlineThickness;
uniform vec3 uOutlineColor;
uniform vec2 uTexelSize;

const vec2 OUTLINE_DIRS[16] = vec2[16](
  vec2(1.0, 0.0), vec2(0.9239, 0.3827), vec2(0.7071, 0.7071), vec2(0.3827, 0.9239),
  vec2(0.0, 1.0), vec2(-0.3827, 0.9239), vec2(-0.7071, 0.7071), vec2(-0.9239, 0.3827),
  vec2(-1.0, 0.0), vec2(-0.9239, -0.3827), vec2(-0.7071, -0.7071), vec2(-0.3827, -0.9239),
  vec2(0.0, -1.0), vec2(0.3827, -0.9239), vec2(0.7071, -0.7071), vec2(0.9239, -0.3827)
);

in vec2 vUV;

out vec4 fragColor;

void main() {
  vec4 scene = texture(uSceneTex, vUV);
  vec4 bloom = texture(uBloomTex, vUV);
  vec3 bloomContrib = max(bloom.rgb - vec3(uThreshold), vec3(0.0));
  vec3 color = scene.rgb + bloomContrib * uIntensity;

  // Outline edge detection (per-entity ID based, with distance scaling)
  if (uOutlineThickness > 0.0) {
    vec4 centerSample = texture(uOutlineTex, vUV);
    float centerVal = centerSample.r;
    float centerScale = centerSample.g;

    // Probe 4 cardinal neighbors at half-thickness to find nearby outline scale
    vec2 pr = uOutlineThickness * 0.5 * uTexelSize;
    float p0 = texture(uOutlineTex, vUV + vec2(pr.x, 0.0)).g;
    float p1 = texture(uOutlineTex, vUV - vec2(pr.x, 0.0)).g;
    float p2 = texture(uOutlineTex, vUV + vec2(0.0, pr.y)).g;
    float p3 = texture(uOutlineTex, vUV - vec2(0.0, pr.y)).g;
    float probeScale = max(max(p0, p1), max(p2, p3));

    // Use whichever scale is larger — ensures closer object's outline dominates at boundaries
    float effectiveScale = max(centerScale, probeScale);

    float scaledThickness = uOutlineThickness * effectiveScale;
    float isOutline = 0.0;
    for (int i = 0; i < 16; i++) {
      vec2 offset = OUTLINE_DIRS[i] * uTexelSize * scaledThickness;
      float neighborVal = texture(uOutlineTex, vUV + offset).r;
      float eitherOutlined = step(0.002, max(centerVal, neighborVal));
      float diff = abs(centerVal - neighborVal);
      float maxId = max(centerVal, neighborVal);
      float relDiff = diff / max(maxId, 0.001);
      float different = step(0.3, relDiff);
      isOutline = max(isOutline, eitherOutlined * different);
    }
    color = mix(color, uOutlineColor, isOutline);
  }

  fragColor = vec4(color, scene.a);
}
`
