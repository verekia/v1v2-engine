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
};
@group(2) @binding(0) var<uniform> lighting : Lighting;

struct VSOut {
  @builtin(position) pos       : vec4f,
  @location(0)       worldNorm : vec3f,
  @location(1)       vertColor : vec3f,
};

@fragment fn fs(input : VSOut) -> @location(0) vec4f {
  let N = normalize(input.worldNorm);
  let L = normalize(-lighting.direction.xyz);
  let NdotL = max(dot(N, L), 0.0);
  let diffuse  = lighting.dirColor.rgb * NdotL;
  let ambient  = lighting.ambientColor.rgb;
  let finalColor = model.color.rgb * input.vertColor * (diffuse + ambient);
  return vec4f(finalColor, 1.0);
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
  out.worldNorm = (model.world * vec4f(input.normal, 0.0)).xyz;
  out.vertColor = input.color;
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
  out.worldNorm = (model.world * skinMat * vec4f(input.normal, 0.0)).xyz;
  out.vertColor = input.color;
  return out;
}
`
