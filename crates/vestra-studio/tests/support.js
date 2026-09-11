'use strict';
// Loads the browser script exactly as Studio serves it, in an isolated VM
// context, so the tests exercise the shipped file rather than a re-export.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function loadCameraControls() {
  const context = {};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(`${__dirname}/../src/camera-controls.js`, 'utf8'), context);
  return context.VestraCameraControls;
}

function close(actual, expected, tolerance = 1e-9, message = '') {
  assert.equal(actual.length, expected.length, `length mismatch ${message}`);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < tolerance, `${message} [${actual}] != [${expected}] at ${index}`));
}

// Structural equality across VM realms: values built inside the script's
// context have their own Array/Object prototypes, which strict deep equality
// would (correctly, but unhelpfully) reject.
function same(actual, expected, message) {
  assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected, message);
}

// Minimal stand-in for a DOM KeyboardEvent: only the fields the controller
// reads. `code` defaults to the physical key for the printed character.
const CODES = { w: 'KeyW', a: 'KeyA', s: 'KeyS', d: 'KeyD', r: 'KeyR', e: 'KeyE', x: 'KeyX', ArrowUp: 'ArrowUp', ArrowDown: 'ArrowDown', ArrowLeft: 'ArrowLeft', ArrowRight: 'ArrowRight', Meta: 'MetaLeft', Control: 'ControlLeft', Alt: 'AltLeft', Shift: 'ShiftLeft', Escape: 'Escape' };
function keyEvent(key, overrides = {}) {
  const base = key.length === 1 ? key.toLowerCase() : key;
  return { key, code: CODES[base] || '', repeat: false, metaKey: false, ctrlKey: false, altKey: false, shiftKey: false, isComposing: false, ...overrides };
}

// Camera-space coordinates of a world point: +X right on screen, +Y up on
// screen, -Z in front of the camera. This is what the rendered image shows.
function project(controls, eye, orientation, point) {
  const matrix = controls.viewMatrix(eye, orientation);
  return [0, 1, 2].map(row => matrix[row] * point[0] + matrix[4 + row] * point[1] + matrix[8 + row] * point[2] + matrix[12 + row]);
}

module.exports = { assert, close, keyEvent, loadCameraControls, project, same };
