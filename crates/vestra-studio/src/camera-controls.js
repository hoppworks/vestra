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
  // Keyboard navigation deliberately has no roll. Yaw is unbounded, while
  // pitch is clamped before the camera can cross its vertical singularity.
  function lookOrientation(yaw, pitch) {
    const maxPitch = Math.PI / 2 - .02;
    const safePitch = Math.max(-maxPitch, Math.min(maxPitch, pitch));
    return normalizeQuaternion(quaternionMultiply(axisAngle([0, 1, 0], yaw), axisAngle([1, 0, 0], safePitch)));
  }
  // backward points target → camera, so forward is -backward.
  function cameraBasis(orientation) { const backward = normalize(rotate(orientation, [0, 0, 1])), rotatedUp = normalize(rotate(orientation, [0, 1, 0])), right = normalize(cross(rotatedUp, backward)), up = normalize(cross(backward, right)); return { right, up, backward, forward: backward.map(value => -value) }; }
  // Column-major WebGL view matrix for a first-person camera. The eye is a
  // position, never an orbit target: changing orientation cannot translate it.
  function viewMatrix(eye, orientation) {
    const { right, up, backward } = cameraBasis(orientation);
    return [right[0], up[0], backward[0], 0, right[1], up[1], backward[1], 0, right[2], up[2], backward[2], 0, -right.reduce((sum, value, axis) => sum + value * eye[axis], 0), -up.reduce((sum, value, axis) => sum + value * eye[axis], 0), -backward.reduce((sum, value, axis) => sum + value * eye[axis], 0), 1];
  }
  // Builds a camera orientation from an orthonormal world-space basis.
  function orientationFromBasis(right, up, backward) {
    const r = normalize(right), u = normalize(up), b = normalize(backward);
    const m00=r[0],m01=u[0],m02=b[0],m10=r[1],m11=u[1],m12=b[1],m20=r[2],m21=u[2],m22=b[2],trace=m00+m11+m22;
    let q;
    if(trace>0){const s=Math.sqrt(trace+1)*2;q=[(m21-m12)/s,(m02-m20)/s,(m10-m01)/s,.25*s];}
    else if(m00>m11&&m00>m22){const s=Math.sqrt(1+m00-m11-m22)*2;q=[.25*s,(m01+m10)/s,(m02+m20)/s,(m21-m12)/s];}
    else if(m11>m22){const s=Math.sqrt(1+m11-m00-m22)*2;q=[(m01+m10)/s,.25*s,(m12+m21)/s,(m02-m20)/s];}
    else{const s=Math.sqrt(1+m22-m00-m11)*2;q=[(m02+m20)/s,(m12+m21)/s,.25*s,(m10-m01)/s];}
    return normalizeQuaternion(q);
  }
  function move(center, orientation, distance, command, ratio = .06) { const basis = cameraBasis(orientation), step = Math.max(distance * ratio, .01), direction = {forward:basis.forward, backward:basis.backward, left:basis.right.map(value => -value), right:basis.right, up:[0,1,0], down:[0,-1,0]}[command]; return direction ? center.map((value, axis) => value + direction[axis] * step) : center.slice(); }
  function commandForKey(key) { return {ArrowUp:'forward',ArrowDown:'backward',ArrowLeft:'left',ArrowRight:'right',w:'lookUp',W:'lookUp',s:'lookDown',S:'lookDown',a:'lookLeft',A:'lookLeft',d:'lookRight',D:'lookRight'}[key] || null; }
  // DA3 camera coordinates are +X right, +Y down, +Z forward. WebGL's scene
  // presentation uses +X right, +Y up, -Z forward. This proper 180° X rotation
  // keeps positions, normals, and camera rays coherent.
  function cameraToViewer(vector) { return [vector[0], -vector[1], -vector[2]]; }
  root.VestraCameraControls = { axisAngle, cameraBasis, cameraToViewer, commandForKey, lookOrientation, move, normalizeQuaternion, orbit, orientationFromBasis, quaternionMultiply, rotate, viewMatrix };
}(typeof window === 'undefined' ? globalThis : window));
