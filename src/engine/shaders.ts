// Shared WGSL structs and fragment shader
const sharedStructs = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
};
@group(1) @binding(0) var<uniform> model : Model;

struct Lighting {
  direction    : vec4f,
  dirColor     : vec4f,
  ambientColor : vec4f,
  lightVP      : mat4x4f,
  shadowParams : vec4f,
};
@group(2) @binding(0) var<uniform> lighting : Lighting;
@group(2) @binding(1) var shadowMap : texture_depth_2d;
@group(2) @binding(2) var shadowSampler : sampler_comparison;

struct VSOut {
  @builtin(position) pos        : vec4f,
  @location(0)       worldNorm  : vec3f,
  @location(1)       vertColor  : vec3f,
  @location(2)       shadowCoord : vec3f,
};

// Poisson disk samples (9 taps)
const POISSON_DISK = array<vec2f, 9>(
  vec2f(-0.7071,  0.7071),
  vec2f( 0.0,    -0.8750),
  vec2f( 0.5303,  0.5303),
  vec2f(-0.6250,  0.0),
  vec2f( 0.8660, -0.25),
  vec2f(-0.25,   -0.4330),
  vec2f( 0.3536,  0.3536),
  vec2f(-0.4330,  0.25),
  vec2f( 0.125,  -0.2165),
);

fn pcfShadow(coord : vec3f) -> f32 {
  let bias = lighting.shadowParams.x;
  let texelSize = lighting.shadowParams.z;
  let enabled = lighting.shadowParams.w;
  // Sample unconditionally (textureSampleCompare requires uniform control flow)
  let refDepth = coord.z - bias;
  // Clamp UV to valid range for sampling (out-of-bounds handled after)
  let uv = clamp(coord.xy, vec2f(0.0), vec2f(1.0));
  // Per-pixel pseudo-random rotation from fragment position
  let rnd = fract(sin(dot(coord.xy, vec2f(12.9898, 78.233))) * 43758.5453);
  let angle = rnd * 6.2831853;
  let cosA = cos(angle);
  let sinA = sin(angle);
  let rotMat = mat2x2f(cosA, sinA, -sinA, cosA);
  var shadow = 0.0;
  let spread = texelSize * 1.0;
  for (var i = 0; i < 9; i++) {
    let offset = rotMat * POISSON_DISK[i] * spread;
    shadow += textureSampleCompare(shadowMap, shadowSampler, uv + offset, refDepth);
  }
  shadow /= 9.0;
  // Outside shadow map (XY or Z) or disabled → fully lit
  let inBounds = step(0.0, coord.x) * step(coord.x, 1.0) * step(0.0, coord.y) * step(coord.y, 1.0) * step(0.0, coord.z) * step(coord.z, 1.0);
  return mix(1.0, shadow, inBounds * step(0.5, enabled));
}

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  let N = normalize(input.worldNorm);
  let L = normalize(-lighting.direction.xyz);
  let NdotL = max(dot(N, L), 0.0);
  let shadow = pcfShadow(input.shadowCoord);
  let diffuse  = lighting.dirColor.rgb * NdotL * shadow;
  let ambient  = lighting.ambientColor.rgb;
  let finalColor = model.color.rgb * input.vertColor * (diffuse + ambient);
  return vec4f(finalColor, model.color.a);
}
`

export const lambertShader = /* wgsl */ `
${sharedStructs}

struct VSIn {
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
  @location(2) color    : vec3f,
};

@vertex fn vs(input : VSIn) -> VSOut {
  var out : VSOut;
  let worldPos = model.world * vec4f(input.position, 1.0);
  out.pos = camera.projection * camera.view * worldPos;
  let worldNorm = (model.world * vec4f(input.normal, 0.0)).xyz;
  out.worldNorm = worldNorm;
  out.vertColor = input.color;
  // Shadow coord: offset along normal to reduce light bleeding at contact edges
  let normalBias = lighting.shadowParams.y;
  let shadowPos = worldPos.xyz + normalize(worldNorm) * normalBias;
  let lightClip = lighting.lightVP * vec4f(shadowPos, 1.0);
  out.shadowCoord = vec3f(
    lightClip.x * 0.5 + 0.5,
    1.0 - (lightClip.y * 0.5 + 0.5),
    lightClip.z,
  );
  return out;
}
`

export const skinnedLambertShader = /* wgsl */ `
${sharedStructs}

@group(3) @binding(0) var<storage, read> jointMatrices : array<mat4x4f>;

struct VSIn {
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
  @location(2) color    : vec3f,
  @location(3) joints   : vec4u,
  @location(4) weights  : vec4f,
};

@vertex fn vs(input : VSIn) -> VSOut {
  var out : VSOut;

  let skinMat =
    input.weights.x * jointMatrices[input.joints.x] +
    input.weights.y * jointMatrices[input.joints.y] +
    input.weights.z * jointMatrices[input.joints.z] +
    input.weights.w * jointMatrices[input.joints.w];

  let worldPos = model.world * skinMat * vec4f(input.position, 1.0);
  out.pos = camera.projection * camera.view * worldPos;
  let worldNorm = (model.world * skinMat * vec4f(input.normal, 0.0)).xyz;
  out.worldNorm = worldNorm;
  out.vertColor = input.color;
  let normalBias = lighting.shadowParams.y;
  let shadowPos = worldPos.xyz + normalize(worldNorm) * normalBias;
  let lightClip = lighting.lightVP * vec4f(shadowPos, 1.0);
  out.shadowCoord = vec3f(
    lightClip.x * 0.5 + 0.5,
    1.0 - (lightClip.y * 0.5 + 0.5),
    lightClip.z,
  );
  return out;
}
`

// ── Textured Lambert shader (AO map) ──────────────────────────────────

const texturedSharedStructs = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
};
@group(1) @binding(0) var<uniform> model : Model;

struct Lighting {
  direction    : vec4f,
  dirColor     : vec4f,
  ambientColor : vec4f,
  lightVP      : mat4x4f,
  shadowParams : vec4f,
};
@group(2) @binding(0) var<uniform> lighting : Lighting;
@group(2) @binding(1) var shadowMap : texture_depth_2d;
@group(2) @binding(2) var shadowSampler : sampler_comparison;

@group(3) @binding(0) var aoTexture : texture_2d<f32>;
@group(3) @binding(1) var aoSampler : sampler;

struct VSOut {
  @builtin(position) pos        : vec4f,
  @location(0)       worldNorm  : vec3f,
  @location(1)       vertColor  : vec3f,
  @location(2)       shadowCoord : vec3f,
  @location(3)       vUV        : vec2f,
};

// Poisson disk samples (9 taps)
const POISSON_DISK = array<vec2f, 9>(
  vec2f(-0.7071,  0.7071),
  vec2f( 0.0,    -0.8750),
  vec2f( 0.5303,  0.5303),
  vec2f(-0.6250,  0.0),
  vec2f( 0.8660, -0.25),
  vec2f(-0.25,   -0.4330),
  vec2f( 0.3536,  0.3536),
  vec2f(-0.4330,  0.25),
  vec2f( 0.125,  -0.2165),
);

fn pcfShadow(coord : vec3f) -> f32 {
  let bias = lighting.shadowParams.x;
  let texelSize = lighting.shadowParams.z;
  let enabled = lighting.shadowParams.w;
  let refDepth = coord.z - bias;
  let uv = clamp(coord.xy, vec2f(0.0), vec2f(1.0));
  let rnd = fract(sin(dot(coord.xy, vec2f(12.9898, 78.233))) * 43758.5453);
  let angle = rnd * 6.2831853;
  let cosA = cos(angle);
  let sinA = sin(angle);
  let rotMat = mat2x2f(cosA, sinA, -sinA, cosA);
  var shadow = 0.0;
  let spread = texelSize * 1.0;
  for (var i = 0; i < 9; i++) {
    let offset = rotMat * POISSON_DISK[i] * spread;
    shadow += textureSampleCompare(shadowMap, shadowSampler, uv + offset, refDepth);
  }
  shadow /= 9.0;
  let inBounds = step(0.0, coord.x) * step(coord.x, 1.0) * step(0.0, coord.y) * step(coord.y, 1.0) * step(0.0, coord.z) * step(coord.z, 1.0);
  return mix(1.0, shadow, inBounds * step(0.5, enabled));
}

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  let N = normalize(input.worldNorm);
  let L = normalize(-lighting.direction.xyz);
  let NdotL = max(dot(N, L), 0.0);
  let shadow = pcfShadow(input.shadowCoord);
  let diffuse  = lighting.dirColor.rgb * NdotL * shadow;
  let ao = textureSample(aoTexture, aoSampler, input.vUV).r;
  let ambient  = lighting.ambientColor.rgb * ao;
  let finalColor = model.color.rgb * input.vertColor * (diffuse + ambient);
  return vec4f(finalColor, model.color.a);
}
`

export const texturedLambertShader = /* wgsl */ `
${texturedSharedStructs}

struct VSIn {
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
  @location(2) color    : vec3f,
  @location(3) uv       : vec2f,
};

@vertex fn vs(input : VSIn) -> VSOut {
  var out : VSOut;
  let worldPos = model.world * vec4f(input.position, 1.0);
  out.pos = camera.projection * camera.view * worldPos;
  let worldNorm = (model.world * vec4f(input.normal, 0.0)).xyz;
  out.worldNorm = worldNorm;
  out.vertColor = input.color;
  out.vUV = input.uv;
  let normalBias = lighting.shadowParams.y;
  let shadowPos = worldPos.xyz + normalize(worldNorm) * normalBias;
  let lightClip = lighting.lightVP * vec4f(shadowPos, 1.0);
  out.shadowCoord = vec3f(
    lightClip.x * 0.5 + 0.5,
    1.0 - (lightClip.y * 0.5 + 0.5),
    lightClip.z,
  );
  return out;
}
`

// ── Unlit shader (no lighting, no shadows) ────────────────────────────

export const unlitShader = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
};
@group(1) @binding(0) var<uniform> model : Model;

struct VSOut {
  @builtin(position) pos       : vec4f,
  @location(0)       vertColor : vec3f,
};

struct VSIn {
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
  @location(2) color    : vec3f,
};

@vertex fn vs(input : VSIn) -> VSOut {
  var out : VSOut;
  let worldPos = model.world * vec4f(input.position, 1.0);
  out.pos = camera.projection * camera.view * worldPos;
  out.vertColor = input.color;
  return out;
}

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  return vec4f(model.color.rgb * input.vertColor, model.color.a);
}
`

// ── Shadow depth shaders (depth-only, no fragment stage) ──────────────

export const shadowDepthShader = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
};
@group(1) @binding(0) var<uniform> model : Model;

@vertex fn vs(@location(0) position : vec3f) -> @builtin(position) vec4f {
  let worldPos = model.world * vec4f(position, 1.0);
  return camera.projection * camera.view * worldPos;
}
`

export const skinnedShadowDepthShader = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
};
@group(1) @binding(0) var<uniform> model : Model;

@group(2) @binding(0) var<storage, read> jointMatrices : array<mat4x4f>;

@vertex fn vs(
  @location(0) position : vec3f,
  @location(3) joints   : vec4u,
  @location(4) weights  : vec4f,
) -> @builtin(position) vec4f {
  let skinMat =
    weights.x * jointMatrices[joints.x] +
    weights.y * jointMatrices[joints.y] +
    weights.z * jointMatrices[joints.z] +
    weights.w * jointMatrices[joints.w];

  let worldPos = model.world * skinMat * vec4f(position, 1.0);
  return camera.projection * camera.view * worldPos;
}
`
