export const lambertShader = /* wgsl */ `

// ── Bind groups ───────────────────────────────────────────────────────
// Group 0: Camera
struct Camera {
  view       : mat4x4f,
  projection : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

// Group 1: Model (dynamic offset)
struct Model {
  world : mat4x4f,
  color : vec4f,
};
@group(1) @binding(0) var<uniform> model : Model;

// Group 2: Lighting
struct Lighting {
  direction    : vec4f,
  dirColor     : vec4f,
  ambientColor : vec4f,
};
@group(2) @binding(0) var<uniform> lighting : Lighting;

// ── Vertex ────────────────────────────────────────────────────────────
struct VSIn {
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
  @location(2) color    : vec3f,
};

struct VSOut {
  @builtin(position) pos       : vec4f,
  @location(0)       worldNorm : vec3f,
  @location(1)       vertColor : vec3f,
};

@vertex fn vs(input : VSIn) -> VSOut {
  var out : VSOut;
  let worldPos = model.world * vec4f(input.position, 1.0);
  out.pos = camera.projection * camera.view * worldPos;
  out.worldNorm = (model.world * vec4f(input.normal, 0.0)).xyz;
  out.vertColor = input.color;
  return out;
}

// ── Fragment ──────────────────────────────────────────────────────────
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
