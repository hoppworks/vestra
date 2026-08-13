/* Pure camera movement math for Vestra Studio; independently testable. */
(function (root) {
  'use strict';
  function normalize(vector) { const length = Math.hypot(...vector) || 1; return vector.map(value => value / length); }
  function cross(left, right) { return [left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0]]; }
  function quaternionMultiply(left, right) { return [left[3] * right[0] + left[0] * right[3] + left[1] * right[2] - left[2] * right[1], left[3] * right[1] - left[0] * right[2] + left[1] * right[3] + left[2] * right[0], left[3] * right[2] + left[0] * right[1] - left[1] * right[0] + left[2] * right[3], left[3] * right[3] - left[0] * right[0] - left[1] * right[1] - left[2] * right[2]]; }
  function normalizeQuaternion(quaternion) { const length = Math.hypot(...quaternion); return Number.isFinite(length) && length > 1e-12 ? quaternion.map(value => value / length) : [0, 0, 0, 1]; }
  function axisAngle(axis, angle) { const sine = Math.sin(angle / 2), length = Math.hypot(...axis) || 1; return [axis[0] / length * sine, axis[1] / length * sine, axis[2] / length * sine, Math.cos(angle / 2)]; }
  function rotate(quaternion, vector) { const unit = normalizeQuaternion(quaternion); return quaternionMultiply(quaternionMultiply(unit, [vector[0], vector[1], vector[2], 0]), [-unit[0], -unit[1], -unit[2], unit[3]]).slice(0, 3); }
  function orbit(orientation, horizontal, vertical) { let next = normalizeQuaternion(quaternionMultiply(axisAngle([0, 1, 0], horizontal), orientation)); const right = rotate(next, [1, 0, 0]); return normalizeQuaternion(quaternionMultiply(axisAngle(right, vertical), next)); }
  // backward points target → camera, so forward is -backward.
  function cameraBasis(orientation) { const backward = normalize(rotate(orientation, [0, 0, 1])), rotatedUp = normalize(rotate(orientation, [0, 1, 0])), right = normalize(cross(rotatedUp, backward)), up = normalize(cross(backward, right)); return { right, up, backward, forward: backward.map(value => -value) }; }
  function move(center, orientation, distance, command, ratio = .06) { const basis = cameraBasis(orientation), step = Math.max(distance * ratio, .01), direction = {forward:basis.forward, backward:basis.backward, left:basis.right.map(value => -value), right:basis.right, up:[0,1,0], down:[0,-1,0]}[command]; return direction ? center.map((value, axis) => value + direction[axis] * step) : center.slice(); }
  function commandForKey(key) { return {ArrowUp:'forward',ArrowDown:'backward',ArrowLeft:'left',ArrowRight:'right',e:'up',E:'up',d:'down',D:'down'}[key] || null; }
  // DA3 camera coordinates are +X right, +Y down, +Z forward. WebGL's scene
  // presentation uses +X right, +Y up, -Z forward. This proper 180° X rotation
  // keeps positions, normals, and camera rays coherent.
  function cameraToViewer(vector) { return [vector[0], -vector[1], -vector[2]]; }
  root.VestraCameraControls = { axisAngle, cameraBasis, cameraToViewer, commandForKey, move, normalizeQuaternion, orbit, quaternionMultiply, rotate };
}(typeof window === 'undefined' ? globalThis : window));
