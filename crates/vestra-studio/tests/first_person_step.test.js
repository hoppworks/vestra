'use strict';
// Frame stepping for the keyboard camera: what the labelled keys do on
// screen, and that the result depends on time, not on frame rate.
const { test } = require('node:test');
const { assert, close, loadCameraControls, project } = require('./support');

const controls = loadCameraControls();
const rates = { move: 2, turn: 1.3 };
const rest = () => ({ eye: [0, 0, 0], yaw: 0, pitch: 0 });
const forwardOf = state => controls.cameraBasis(state.orientation).forward;
const step = (state, commands, seconds = 1 / 60, speeds = rates) => controls.stepFirstPerson(state, commands, seconds, speeds);
const hold = (state, commands, seconds, frames, speeds = rates) => { let next = state; for (let index = 0; index < frames; index++) next = step(next, commands, seconds / frames, speeds); return next; };

test('look keys turn toward the labelled side of the screen', () => {
  const left = step(rest(), ['lookLeft']), right = step(rest(), ['lookRight']), up = step(rest(), ['lookUp']), down = step(rest(), ['lookDown']);
  assert.ok(forwardOf(left)[0] < 0, `lookLeft must swing the view toward -X, got ${forwardOf(left)}`);
  assert.ok(forwardOf(right)[0] > 0, `lookRight must swing the view toward +X, got ${forwardOf(right)}`);
  assert.ok(forwardOf(up)[1] > 0, `lookUp must raise the view, got ${forwardOf(up)}`);
  assert.ok(forwardOf(down)[1] < 0, `lookDown must lower the view, got ${forwardOf(down)}`);
  // What the user sees: after turning right, a landmark that sat on the right
  // edge of the image moves toward the centre; after looking up, a landmark
  // above the centre moves down toward it.
  const landmarkRight = [1, 0, -2], landmarkAbove = [0, 1, -2];
  const before = project(controls, [0, 0, 0], controls.lookOrientation(0, 0), landmarkRight);
  const afterRight = project(controls, right.eye, right.orientation, landmarkRight);
  assert.ok(afterRight[0] < before[0], `turning right must bring a right-hand landmark toward the centre: ${before[0]} -> ${afterRight[0]}`);
  const afterLeft = project(controls, left.eye, left.orientation, landmarkRight);
  assert.ok(afterLeft[0] > before[0], 'turning left pushes a right-hand landmark further right');
  const aboveBefore = project(controls, [0, 0, 0], controls.lookOrientation(0, 0), landmarkAbove);
  const aboveAfterUp = project(controls, up.eye, up.orientation, landmarkAbove);
  assert.ok(aboveAfterUp[1] < aboveBefore[1], 'looking up brings an overhead landmark down toward the centre');
  for (const state of [left, right, up, down]) { close(state.eye, [0, 0, 0]); assert.ok(state.turned && !state.moved); }
});

test('movement keys translate relative to the current view', () => {
  close(step(rest(), ['forward'], .5).eye, [0, 0, -1]);
  close(step(rest(), ['backward'], .5).eye, [0, 0, 1]);
  close(step(rest(), ['left'], .5).eye, [-1, 0, 0]);
  close(step(rest(), ['right'], .5).eye, [1, 0, 0]);
  close(step(rest(), ['up'], .5).eye, [0, 1, 0]);
  close(step(rest(), ['down'], .5).eye, [0, -1, 0]);
  // Facing -X after a quarter turn to the left: forward follows the view.
  const facingLeft = { eye: [0, 0, 0], yaw: Math.PI / 2, pitch: 0 };
  close(step(facingLeft, ['forward'], .5).eye, [-1, 0, 0]);
  close(step(facingLeft, ['right'], .5).eye, [0, 0, -1]);
  // Looking down does not sink the camera when strafing, and vertical
  // movement stays on the world axis whatever the pitch.
  const lookingDown = { eye: [0, 0, 0], yaw: 0, pitch: -1 };
  assert.ok(Math.abs(step(lookingDown, ['right'], .5).eye[1]) < 1e-12);
  close(step(lookingDown, ['up'], .5).eye, [0, 1, 0]);
  const moved = step(rest(), ['forward']);
  assert.ok(moved.moved && !moved.turned);
  assert.equal(moved.yaw, 0);
  assert.equal(moved.pitch, 0);
});

test('the same held time produces the same result at any frame rate', () => {
  for (const commands of [['forward'], ['lookLeft'], ['lookUp'], ['forward', 'left'], ['lookLeft', 'lookDown']]) {
    const at30 = hold(rest(), commands, 1, 30), at60 = hold(rest(), commands, 1, 60), at144 = hold(rest(), commands, 1, 144), once = step(rest(), commands, 1);
    close(at30.eye, at60.eye, 1e-6, `${commands} eye 30 vs 60 Hz`);
    close(at144.eye, at60.eye, 1e-6, `${commands} eye 144 vs 60 Hz`);
    close([at30.yaw, at30.pitch], [at144.yaw, at144.pitch], 1e-9, `${commands} angles`);
    close(once.eye, at60.eye, 1e-9, `${commands} single step equals many small steps`);
  }
  // Small worlds move slowly instead of jumping a fixed per-frame minimum.
  const tiny = { move: .001, turn: 1.3 };
  close(step(rest(), ['forward'], 1 / 60, tiny).eye, [0, 0, -.001 / 60]);
  close(hold(rest(), ['forward'], 1, 144, tiny).eye, [0, 0, -.001], 1e-9);
  close(hold(rest(), ['forward'], 1, 30, tiny).eye, [0, 0, -.001], 1e-9);
});

test('turning while moving covers the same distance and angle at any frame rate', () => {
  // The path is an arc integrated per frame, so the exact position may differ
  // slightly between refresh rates; speed and heading must not.
  const distance = (state, commands, frames) => { let travelled = 0, current = state; for (let index = 0; index < frames; index++) { const next = step(current, commands, 1 / frames); travelled += Math.hypot(...next.eye.map((value, axis) => value - current.eye[axis])); current = next; } return { travelled, state: current }; };
  const at30 = distance(rest(), ['backward', 'lookRight'], 30), at144 = distance(rest(), ['backward', 'lookRight'], 144);
  assert.ok(Math.abs(at30.travelled - rates.move) < 1e-9 && Math.abs(at144.travelled - rates.move) < 1e-9, `arc length equals speed × time: ${at30.travelled} / ${at144.travelled}`);
  assert.ok(Math.abs(at30.state.yaw - at144.state.yaw) < 1e-9, 'heading is identical');
  close(at30.state.eye, at144.state.eye, .05, 'positions agree to within the per-frame arc discretization');
});

test('diagonal movement is not faster than a straight line', () => {
  const straight = step(rest(), ['forward'], .25).eye, diagonal = step(rest(), ['forward', 'left'], .25).eye;
  assert.ok(Math.abs(Math.hypot(...straight) - Math.hypot(...diagonal)) < 1e-9, `${Math.hypot(...straight)} vs ${Math.hypot(...diagonal)}`);
  assert.ok(diagonal[0] < 0 && diagonal[2] < 0, 'diagonal still points forward-left');
  const triple = step(rest(), ['forward', 'left', 'up'], .25).eye;
  assert.ok(Math.abs(Math.hypot(...triple) - Math.hypot(...straight)) < 1e-9);
});

test('opposite keys cancel exactly', () => {
  const still = step(rest(), ['forward', 'backward'], .5);
  close(still.eye, [0, 0, 0]);
  assert.ok(!still.moved);
  const steady = step(rest(), ['lookLeft', 'lookRight', 'lookUp', 'lookDown'], .5);
  assert.equal(steady.yaw, 0);
  assert.equal(steady.pitch, 0);
  const sideways = step(rest(), ['forward', 'backward', 'left'], .5);
  close(sideways.eye, [-1, 0, 0], 1e-9, 'the surviving key keeps full speed');
});

test('looking down responds immediately after a long look up', () => {
  const pinned = hold(rest(), ['lookUp'], 100, 600);
  assert.equal(pinned.pitch, controls.MAX_PITCH, 'the stored pitch is clamped, not just the rendered one');
  const released = step(pinned, ['lookDown']);
  assert.ok(released.pitch < pinned.pitch - 1e-6, 'no invisible excess has to be unwound first');
  assert.ok(forwardOf(released)[1] < forwardOf(pinned)[1], 'the view visibly lowers on the very first frame');
  const floor = hold(rest(), ['lookDown'], 100, 600);
  assert.equal(floor.pitch, -controls.MAX_PITCH);
  assert.ok(step(floor, ['lookUp']).pitch > floor.pitch + 1e-6);
});

test('long turns keep a bounded yaw and a unit orientation', () => {
  const spun = hold(rest(), ['lookLeft'], 1000, 4000);
  assert.ok(spun.yaw > -Math.PI && spun.yaw <= Math.PI, `yaw stays wrapped: ${spun.yaw}`);
  assert.ok(Math.abs(Math.hypot(...spun.orientation) - 1) < 1e-9);
  assert.ok(forwardOf(spun).every(Number.isFinite));
  // A full 2π of turning returns to the starting view.
  const around = hold(rest(), ['lookRight'], 2 * Math.PI / rates.turn, 360);
  close(forwardOf(around), [0, 0, -1], 1e-6);
});

test('stepping is pure and ignores invalid time, rates and commands', () => {
  const state = rest(), snapshot = JSON.stringify(state);
  const next = step(state, ['forward', 'lookLeft'], .1);
  assert.equal(JSON.stringify(state), snapshot, 'input state is never mutated');
  assert.notEqual(next.eye, state.eye, 'a fresh eye array is returned');
  for (const seconds of [0, -1, NaN, Infinity, undefined, null, '1']) {
    const frozen = controls.stepFirstPerson(rest(), ['forward', 'lookUp'], seconds, rates);
    close(frozen.eye, [0, 0, 0], 1e-12, `dt=${seconds}`);
    assert.equal(frozen.pitch, 0);
  }
  close(step(rest(), ['forward'], 1, {}).eye, [0, 0, 0], 1e-12, 'missing rates move nothing');
  close(step(rest(), ['forward'], 1, { move: NaN, turn: NaN }).eye, [0, 0, 0], 1e-12);
  const idle = step(rest(), [], 1);
  close(idle.eye, [0, 0, 0]);
  assert.ok(!idle.moved && !idle.turned);
  const nonsense = step(rest(), ['jump', 'sprint'], 1);
  close(nonsense.eye, [0, 0, 0]);
  assert.ok(!nonsense.moved && !nonsense.turned);
  close(step(rest(), ['forward']).orientation, controls.lookOrientation(0, 0));
});

test('a tap impulse is exactly one 60 Hz frame of motion', () => {
  const tap = step(rest(), ['forward'], 1 / 60), frame = step(rest(), ['forward'], 1 / 60);
  close(tap.eye, frame.eye);
  close(tap.eye, [0, 0, -rates.move / 60]);
  const turn = step(rest(), ['lookRight'], 1 / 60);
  assert.ok(Math.abs(turn.yaw + rates.turn / 60) < 1e-12);
});
