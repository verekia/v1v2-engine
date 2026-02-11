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
  bloom : f32,
  bloomWhiten : f32,
  outline : f32,
  outlineScale : f32,
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
  // Outside shadow map (XY or Z) or disabled -> fully lit
  let inBounds = step(0.0, coord.x) * step(coord.x, 1.0) * step(0.0, coord.y) * step(coord.y, 1.0) * step(0.0, coord.z) * step(coord.z, 1.0);
  return mix(1.0, shadow, inBounds * step(0.5, enabled));
}

fn lambertColor(input : VSOut) -> vec4f {
  let N = normalize(input.worldNorm);
  let L = normalize(-lighting.direction.xyz);
  let NdotL = max(dot(N, L), 0.0);
  let shadow = pcfShadow(input.shadowCoord);
  let diffuse  = lighting.dirColor.rgb * NdotL * shadow;
  let ambient  = lighting.ambientColor.rgb;
  let finalColor = model.color.rgb * input.vertColor * (diffuse + ambient);
  return vec4f(finalColor, model.color.a);
}

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  return lambertColor(input);
}

struct FragOutput {
  @location(0) color      : vec4f,
  @location(1) bloomOut   : vec4f,
  @location(2) outlineOut : vec4f,
};

@fragment fn fsMRT(input : VSOut) -> FragOutput {
  let c = lambertColor(input);
  var out : FragOutput;
  out.color = vec4f(mix(c.rgb, vec3f(1.0), model.bloomWhiten), c.a);
  out.bloomOut = vec4f(c.rgb * model.bloom, c.a);
  out.outlineOut = vec4f(model.outline, model.outlineScale, 0.0, 1.0);
  return out;
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

// -- Textured Lambert shader (AO map) --

const texturedSharedStructs = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
  bloom : f32,
  bloomWhiten : f32,
  outline : f32,
  outlineScale : f32,
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

fn texturedLambertColor(input : VSOut) -> vec4f {
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

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  return texturedLambertColor(input);
}

struct FragOutput {
  @location(0) color      : vec4f,
  @location(1) bloomOut   : vec4f,
  @location(2) outlineOut : vec4f,
};

@fragment fn fsMRT(input : VSOut) -> FragOutput {
  let c = texturedLambertColor(input);
  var out : FragOutput;
  out.color = vec4f(mix(c.rgb, vec3f(1.0), model.bloomWhiten), c.a);
  out.bloomOut = vec4f(c.rgb * model.bloom, c.a);
  out.outlineOut = vec4f(model.outline, model.outlineScale, 0.0, 1.0);
  return out;
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

// -- Unlit shader (no lighting, no shadows) --

export const unlitShader = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
  bloom : f32,
  bloomWhiten : f32,
  outline : f32,
  outlineScale : f32,
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

fn unlitColor(input : VSOut) -> vec4f {
  return vec4f(model.color.rgb * input.vertColor, model.color.a);
}

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  return unlitColor(input);
}

struct FragOutput {
  @location(0) color      : vec4f,
  @location(1) bloomOut   : vec4f,
  @location(2) outlineOut : vec4f,
};

@fragment fn fsMRT(input : VSOut) -> FragOutput {
  let c = unlitColor(input);
  var out : FragOutput;
  out.color = vec4f(mix(c.rgb, vec3f(1.0), model.bloomWhiten), c.a);
  out.bloomOut = vec4f(c.rgb * model.bloom, c.a);
  out.outlineOut = vec4f(model.outline, model.outlineScale, 0.0, 1.0);
  return out;
}
`

// -- Shadow depth shaders (depth-only, no fragment stage) --

export const shadowDepthShader = /* wgsl */ `
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Model {
  world : mat4x4f,
  color : vec4f,
  bloom : f32,
  bloomWhiten : f32,
  outline : f32,
  outlineScale : f32,
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
  bloom : f32,
  bloomWhiten : f32,
  outline : f32,
  outlineScale : f32,
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

// -- Bloom post-processing shaders --

export const bloomDownsampleShader = /* wgsl */ `
struct Params {
  srcTexelSize : vec2f,
};
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var<uniform> params : Params;

@vertex fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  return vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos : vec4f) -> @location(0) vec4f {
  // pos.xy is in dest pixels; dest = src / 2, so uv = pos.xy * srcTexelSize * 2
  let uv = pos.xy * params.srcTexelSize * 2.0;
  let t = params.srcTexelSize;
  // 4-tap bilinear downsample (each tap averages a 2x2 block via linear filtering)
  let a = textureSample(src, samp, uv + vec2(-t.x, -t.y) * 0.5);
  let b = textureSample(src, samp, uv + vec2( t.x, -t.y) * 0.5);
  let c = textureSample(src, samp, uv + vec2(-t.x,  t.y) * 0.5);
  let d = textureSample(src, samp, uv + vec2( t.x,  t.y) * 0.5);
  return (a + b + c + d) * 0.25;
}
`

export const bloomUpsampleShader = /* wgsl */ `
struct Params {
  destTexelSize : vec2f,
  offset        : vec2f, // precomputed: filterRadius * srcTexelSize
};
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var<uniform> params : Params;

@vertex fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  return vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos : vec4f) -> @location(0) vec4f {
  let uv = pos.xy * params.destTexelSize;
  let o = params.offset;
  // 9-tap tent filter
  var color = textureSample(src, samp, uv) * 4.0;
  color += textureSample(src, samp, uv + vec2(-o.x, 0.0)) * 2.0;
  color += textureSample(src, samp, uv + vec2( o.x, 0.0)) * 2.0;
  color += textureSample(src, samp, uv + vec2(0.0, -o.y)) * 2.0;
  color += textureSample(src, samp, uv + vec2(0.0,  o.y)) * 2.0;
  color += textureSample(src, samp, uv + vec2(-o.x, -o.y));
  color += textureSample(src, samp, uv + vec2( o.x, -o.y));
  color += textureSample(src, samp, uv + vec2(-o.x,  o.y));
  color += textureSample(src, samp, uv + vec2( o.x,  o.y));
  return color / 16.0;
}
`

export const bloomCompositeShader = /* wgsl */ `
struct CompositeParams {
  intensity        : f32,
  threshold        : f32,
  outlineThickness : f32,
  _pad0            : f32,
  outlineColor     : vec4f,
  texelSize        : vec2f,
  _pad1            : vec2f,
};
@group(0) @binding(0) var sceneTex : texture_2d<f32>;
@group(0) @binding(1) var bloomTex : texture_2d<f32>;
@group(0) @binding(2) var outlineTex : texture_2d<f32>;
@group(0) @binding(3) var texSampler : sampler;
@group(0) @binding(4) var<uniform> params : CompositeParams;

const OUTLINE_DIRS = array<vec2f, 16>(
  vec2f(1.0, 0.0), vec2f(0.9239, 0.3827), vec2f(0.7071, 0.7071), vec2f(0.3827, 0.9239),
  vec2f(0.0, 1.0), vec2f(-0.3827, 0.9239), vec2f(-0.7071, 0.7071), vec2f(-0.9239, 0.3827),
  vec2f(-1.0, 0.0), vec2f(-0.9239, -0.3827), vec2f(-0.7071, -0.7071), vec2f(-0.3827, -0.9239),
  vec2f(0.0, -1.0), vec2f(0.3827, -0.9239), vec2f(0.7071, -0.7071), vec2f(0.9239, -0.3827),
);

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
};

@vertex fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  var out : VSOut;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.pos = vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(x, 1.0 - y);
  return out;
}

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  let scene = textureSample(sceneTex, texSampler, input.uv);
  let bloom = textureSample(bloomTex, texSampler, input.uv);
  let bloomContrib = max(bloom.rgb - vec3f(params.threshold), vec3f(0.0));
  var color = scene.rgb + bloomContrib * params.intensity;

  // Outline edge detection (per-entity ID based, with distance scaling)
  if (params.outlineThickness > 0.0) {
    let centerSample = textureSample(outlineTex, texSampler, input.uv);
    let centerVal = centerSample.r;
    let centerScale = centerSample.g;

    // Probe 4 cardinal neighbors at half-thickness to find nearby outline scale
    let pr = params.outlineThickness * 0.5 * params.texelSize;
    let p0 = textureSample(outlineTex, texSampler, input.uv + vec2f(pr.x, 0.0)).g;
    let p1 = textureSample(outlineTex, texSampler, input.uv - vec2f(pr.x, 0.0)).g;
    let p2 = textureSample(outlineTex, texSampler, input.uv + vec2f(0.0, pr.y)).g;
    let p3 = textureSample(outlineTex, texSampler, input.uv - vec2f(0.0, pr.y)).g;
    let probeScale = max(max(p0, p1), max(p2, p3));

    // Use whichever scale is larger — ensures closer object's outline dominates at boundaries
    let effectiveScale = max(centerScale, probeScale);

    let scaledThickness = params.outlineThickness * effectiveScale;
    var isOutline = 0.0;
    for (var i = 0u; i < 16u; i++) {
      let offset = OUTLINE_DIRS[i] * params.texelSize * scaledThickness;
      let neighborVal = textureSample(outlineTex, texSampler, input.uv + offset).r;
      let eitherOutlined = step(0.002, max(centerVal, neighborVal));
      let diff = abs(centerVal - neighborVal);
      let maxId = max(centerVal, neighborVal);
      let relDiff = diff / max(maxId, 0.001);
      let different = step(0.3, relDiff);
      isOutline = max(isOutline, eitherOutlined * different);
    }
    color = mix(color, params.outlineColor.rgb, isOutline);
  }

  return vec4f(color, scene.a);
}
`
