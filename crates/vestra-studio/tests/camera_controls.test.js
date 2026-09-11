'use strict';
const { test } = require('node:test');
const { assert, close, loadCameraControls, project } = require('./support');

const controls = loadCameraControls();
const identity = [0, 0, 0, 1];
const unit = quaternion => Math.abs(Math.hypot(...quaternion) - 1) < 1e-12;

test('move translates relative to the current view', () => {
  close(controls.move([0, 0, 0], identity, 10, 'forward'), [0, 0, -.6]);
  close(controls.move([0, 0, 0], identity, 10, 'backward'), [0, 0, .6]);
  close(controls.move([0, 0, 0], identity, 10, 'left'), [-.6, 0, 0]);
  close(controls.move([0, 0, 0], identity, 10, 'right'), [.6, 0, 0]);
  close(controls.move([1, 2, 3], identity, 10, 'up'), [1, 2.6, 3]);
  close(controls.move([1, 2, 3], identity, 10, 'down'), [1, 1.4, 3]);
  close(controls.move([1, 2, 3], identity, 10, 'nonsense'), [1, 2, 3]);
  // Facing -X after a 90° left turn: forward is -X and "left" is +Z.
  const turnedLeft = controls.lookOrientation(Math.PI / 2, 0);
  close(controls.move([0, 0, 0], turnedLeft, 10, 'forward'), [-.6, 0, 0]);
  close(controls.move([0, 0, 0], turnedLeft, 10, 'left'), [0, 0, .6]);
});

test('move has no minimum step, so small worlds scale linearly too', () => {
  close(controls.move([0, 0, 0], identity, .18, 'forward', .55 / 60), [0, 0, -.18 * .55 / 60]);
  close(controls.move([0, 0, 0], identity, .18, 'forward', .55 / 144), [0, 0, -.18 * .55 / 144]);
});

test('cameraToViewer is the proper 180° X rotation from DA3 to WebGL axes', () => {
  close(controls.cameraToViewer([2, 3, 4]), [2, -3, -4]);
  close(controls.cameraToViewer([0, 1, 0]), [0, -1, 0]);
  close(controls.cameraToViewer([0, 0, 1]), [0, 0, -1]);
});

test('orientationFromBasis round-trips through cameraBasis', () => {
  const matched = controls.cameraBasis(controls.orientationFromBasis([0, 0, 1], [0, 1, 0], [-1, 0, 0]));
  close(matched.right, [0, 0, 1]);
  close(matched.up, [0, 1, 0]);
  close(matched.backward, [-1, 0, 0]);
});

test('orbit stays a unit quaternion with finite axes over long sessions', () => {
  let orientation = identity;
  for (let index = 0; index < 10_000; index++) orientation = controls.orbit(orientation, .013, -.009);
  assert.ok(unit(orientation), 'orbit quaternion remains unit length');
  for (const vector of Object.values(controls.cameraBasis(orientation))) assert.ok(vector.every(Number.isFinite), 'orbit never produces an invalid camera axis');
});

test('look convention: positive yaw turns left, positive pitch looks up', () => {
  const forward = orientation => controls.cameraBasis(orientation).forward;
  close(forward(controls.lookOrientation(0, 0)), [0, 0, -1]);
  const left = forward(controls.lookOrientation(.5, 0)), right = forward(controls.lookOrientation(-.5, 0));
  assert.ok(left[0] < 0 && left[2] < 0, `yaw > 0 must swing forward toward -X (left): ${left}`);
  assert.ok(right[0] > 0 && right[2] < 0, `yaw < 0 must swing forward toward +X (right): ${right}`);
  const up = forward(controls.lookOrientation(0, .5)), down = forward(controls.lookOrientation(0, -.5));
  assert.ok(up[1] > 0, `pitch > 0 must look up: ${up}`);
  assert.ok(down[1] < 0, `pitch < 0 must look down: ${down}`);
  // Screen-space confirmation: a world point on the camera's +X appears on
  // the right of the image, one above appears at the top.
  close(project(controls, [0, 0, 0], identity, [1, 0, -1]), [1, 0, -1]);
  close(project(controls, [0, 0, 0], identity, [0, 1, -1]), [0, 1, -1]);
});

test('first-person navigation rotates the orientation without moving the eye', () => {
  const eye = [7, -2, 11], rotated = controls.lookOrientation(-2.7, .8);
  assert.notDeepEqual(controls.cameraBasis(identity).forward, controls.cameraBasis(rotated).forward);
  close(project(controls, eye, rotated, eye), [0, 0, 0]);
  const forwardPoint = eye.map((value, axis) => value + controls.cameraBasis(rotated).forward[axis] * 5);
  close(project(controls, eye, rotated, forwardPoint), [0, 0, -5]);
});

test('pitch is clamped before the vertical singularity and yaw is unbounded', () => {
  const almostVertical = controls.lookOrientation(17 * Math.PI, Math.PI);
  assert.ok(unit(almostVertical), 'look controller remains normalized after unlimited yaw');
  assert.ok(Math.abs(controls.cameraBasis(almostVertical).forward[1]) < 1, 'pitch clamp prevents an upside-down camera singularity');
  assert.equal(controls.clampPitch(Math.PI), controls.MAX_PITCH);
  assert.equal(controls.clampPitch(-Math.PI), -controls.MAX_PITCH);
  assert.equal(controls.clampPitch(.3), .3);
  assert.ok(controls.MAX_PITCH < Math.PI / 2);
});

test('lookAnglesFromForward inverts lookOrientation up to roll', () => {
  for (const [yaw, pitch] of [[0, 0], [.5, 0], [-2.7, .8], [3, -1.2], [Math.PI, 0], [-1, controls.MAX_PITCH]]) {
    const forward = controls.cameraBasis(controls.lookOrientation(yaw, pitch)).forward;
    const angles = controls.lookAnglesFromForward(forward);
    close(controls.cameraBasis(controls.lookOrientation(angles.yaw, angles.pitch)).forward, forward, 1e-9, `yaw ${yaw} pitch ${pitch}`);
    assert.ok(Math.abs(angles.pitch - pitch) < 1e-9, `pitch round-trips: ${angles.pitch} vs ${pitch}`);
    assert.ok(Math.abs(controls.wrapYaw(angles.yaw - yaw)) < 1e-9, `yaw round-trips: ${angles.yaw} vs ${yaw}`);
  }
  // A rolled basis (camera tilted about its view axis) yields the same
  // heading; keyboard navigation simply continues without the roll.
  const rolled = controls.orientationFromBasis([Math.SQRT1_2, Math.SQRT1_2, 0], [-Math.SQRT1_2, Math.SQRT1_2, 0], [0, 0, 1]);
  const angles = controls.lookAnglesFromForward(controls.cameraBasis(rolled).forward);
  assert.ok(Math.abs(angles.yaw) < 1e-9 && Math.abs(angles.pitch) < 1e-9);
  // Straight up/down never exceeds the clamp or produces NaN.
  const straightUp = controls.lookAnglesFromForward([0, 1, 0]);
  assert.equal(straightUp.pitch, controls.MAX_PITCH);
  assert.ok(Number.isFinite(straightUp.yaw));
  const straightDown = controls.lookAnglesFromForward([0, -1.0000001, 0]);
  assert.equal(straightDown.pitch, -controls.MAX_PITCH);
});

test('wrapYaw keeps the stored angle in (-π, π] without changing the view', () => {
  for (const yaw of [0, 1, -1, 3.5, -3.5, 7 * Math.PI, -13.7]) {
    const wrapped = controls.wrapYaw(yaw);
    assert.ok(wrapped >= -Math.PI && wrapped <= Math.PI, `${yaw} wraps to ${wrapped}`);
    close(controls.cameraBasis(controls.lookOrientation(wrapped, .1)).forward, controls.cameraBasis(controls.lookOrientation(yaw, .1)).forward);
  }
});

test('commandForKey accepts the documented keys in either case and nothing else', () => {
  assert.equal(controls.commandForKey('ArrowUp'), 'forward');
  assert.equal(controls.commandForKey('ArrowDown'), 'backward');
  assert.equal(controls.commandForKey('ArrowLeft'), 'left');
  assert.equal(controls.commandForKey('ArrowRight'), 'right');
  assert.equal(controls.commandForKey('w'), 'lookUp');
  assert.equal(controls.commandForKey('W'), 'lookUp');
  assert.equal(controls.commandForKey('a'), 'lookLeft');
  assert.equal(controls.commandForKey('D'), 'lookRight');
  assert.equal(controls.commandForKey('E'), null);
  assert.equal(controls.commandForKey('x'), null);
  assert.equal(controls.commandForKey(' '), null);
  assert.equal(controls.commandForKey(undefined), null);
});
