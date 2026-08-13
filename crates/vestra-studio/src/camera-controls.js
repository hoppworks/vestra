/* Pure camera movement math for Vestra Studio; independently testable. */
(function (root) {
  'use strict';
  function normalize(vector) { const length = Math.hypot(...vector) || 1; return vector.map(value => value / length); }
  function cross(left, right) { return [left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0]]; }
  function quaternionMultiply(left, right) { return [left[3] * right[0] + left[0] * right[3] + left[1] * right[2] - left[2] * right[1], left[3] * right[1] - left[0] * right[2] + left[1] * right[3] + left[2] * right[0], left[3] * right[2] + left[0] * right[1] - left[1] * right[0] + left[2] * right[3], left[3] * right[3] - left[0] * right[0] - left[1] * right[1] - left[2] * right[2]]; }
  function rotate(quaternion, vector) { return quaternionMultiply(quaternionMultiply(quaternion, [vector[0], vector[1], vector[2], 0]), [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]).slice(0, 3); }
  // backward points target → camera, so forward is -backward.
  function cameraBasis(orientation) { const backward = normalize(rotate(orientation, [0, 0, 1])), rotatedUp = normalize(rotate(orientation, [0, 1, 0])), right = normalize(cross(rotatedUp, backward)), up = normalize(cross(backward, right)); return { right, up, backward, forward: backward.map(value => -value) }; }
  function move(center, orientation, distance, command) { const basis = cameraBasis(orientation), step = Math.max(distance * .06, .01), direction = {forward:basis.forward, backward:basis.backward, left:basis.right.map(value => -value), right:basis.right, up:[0,1,0], down:[0,-1,0]}[command]; return direction ? center.map((value, axis) => value + direction[axis] * step) : center.slice(); }
  function commandForKey(key) { return {ArrowUp:'forward',ArrowDown:'backward',ArrowLeft:'left',ArrowRight:'right',e:'up',E:'up',d:'down',D:'down'}[key] || null; }
  root.VestraCameraControls = { cameraBasis, commandForKey, move };
}(typeof window === 'undefined' ? globalThis : window));
