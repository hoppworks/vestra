'use strict';
// Physical-key tracking for Studio navigation. Every scenario here is a way
// the old Set-of-commands approach could lose or duplicate a key.
const { test } = require('node:test');
const { assert, keyEvent, loadCameraControls, same } = require('./support');

const controls = loadCameraControls();

test('commandForEvent resolves the physical key first and the character second', () => {
  assert.equal(controls.commandForEvent(keyEvent('w')), 'lookUp');
  assert.equal(controls.commandForEvent(keyEvent('W', { shiftKey: true })), 'lookUp');
  assert.equal(controls.commandForEvent(keyEvent('ArrowUp')), 'forward');
  // AZERTY: the cap printed "W" reports code KeyZ; the character still works.
  assert.equal(controls.commandForEvent({ key: 'w', code: 'KeyZ' }), 'lookUp');
  // AZERTY: the physical WASD cluster (code KeyW prints "z") works as well.
  assert.equal(controls.commandForEvent({ key: 'z', code: 'KeyW' }), 'lookUp');
  // Dvorak: the physical KeyW position prints a comma.
  assert.equal(controls.commandForEvent({ key: ',', code: 'KeyW' }), 'lookUp');
  // Virtual keyboards report no code; the character alone must suffice.
  assert.equal(controls.commandForEvent({ key: 'a', code: '' }), 'lookLeft');
  assert.equal(controls.commandForEvent({ key: 'ArrowLeft' }), 'left');
  assert.equal(controls.commandForEvent(keyEvent('e')), null);
  assert.equal(controls.commandForEvent(keyEvent('r')), null);
  assert.equal(controls.commandForEvent(keyEvent('Escape')), null);
  assert.equal(controls.commandForEvent(null), null);
  assert.equal(controls.commandForEvent({}), null);
});

test('commandForEvent ignores IME composition and browser modifier chords', () => {
  assert.equal(controls.commandForEvent(keyEvent('w', { isComposing: true })), null);
  assert.equal(controls.commandForEvent(keyEvent('ArrowLeft', { metaKey: true })), null, '⌘← is browser history, not strafing');
  assert.equal(controls.commandForEvent(keyEvent('ArrowLeft', { ctrlKey: true })), null);
  assert.equal(controls.commandForEvent(keyEvent('w', { altKey: true })), null);
  assert.equal(controls.commandForEvent(keyEvent('w', { shiftKey: true })), 'lookUp', 'Shift is not a browser chord');
  for (const key of ['Meta', 'Control', 'Alt', 'AltGraph', 'OS']) assert.ok(controls.isModifierChord(keyEvent(key)), `${key} itself is a modifier`);
  assert.ok(!controls.isModifierChord(keyEvent('Shift')));
  assert.ok(!controls.isModifierChord(keyEvent('w')));
  assert.ok(!controls.isModifierChord(null));
});

test('keyIdentity is stable for the whole press even when the character changes', () => {
  assert.equal(controls.keyIdentity(keyEvent('w')), controls.keyIdentity(keyEvent('W', { shiftKey: true })));
  assert.equal(controls.keyIdentity(keyEvent('ArrowUp')), 'ArrowUp');
  assert.equal(controls.keyIdentity({ key: 'A', code: '' }), controls.keyIdentity({ key: 'a', code: '' }));
  assert.equal(controls.keyIdentity({ key: 'a', code: 'Unidentified' }), 'key:a');
  assert.equal(controls.keyIdentity({ key: 'ArrowLeft' }), 'key:ArrowLeft');
  assert.notEqual(controls.keyIdentity(keyEvent('w')), controls.keyIdentity(keyEvent('s')));
});

test('a key is held from its first keydown until keyup, auto-repeat included', () => {
  const keyboard = controls.createKeyboardController();
  same(keyboard.press(keyEvent('w')), { command: 'lookUp', fresh: true });
  same(keyboard.commands(), ['lookUp']);
  for (let index = 0; index < 50; index++) same(keyboard.press(keyEvent('w', { repeat: true })), { command: 'lookUp', fresh: false }, 'OS auto-repeat never counts as a new press');
  same(keyboard.commands(), ['lookUp']);
  assert.equal(keyboard.size, 1);
  assert.equal(keyboard.release(keyEvent('w')), 'lookUp');
  same(keyboard.commands(), []);
  assert.equal(keyboard.size, 0);
  // A repeat that arrives without a preceding keydown (focus returned while
  // the key was already down) is still physically held and counts as fresh.
  same(keyboard.press(keyEvent('s', { repeat: true })), { command: 'lookDown', fresh: true });
});

test('several keys are tracked independently and reported once each', () => {
  const keyboard = controls.createKeyboardController();
  keyboard.press(keyEvent('ArrowUp'));
  keyboard.press(keyEvent('ArrowLeft'));
  keyboard.press(keyEvent('d'));
  same(keyboard.commands().sort(), ['forward', 'left', 'lookRight']);
  assert.ok(keyboard.isHeld('forward') && keyboard.isHeld('lookRight') && !keyboard.isHeld('backward'));
  keyboard.release(keyEvent('ArrowLeft'));
  same(keyboard.commands().sort(), ['forward', 'lookRight']);
  // Two physical keys for one command (AZERTY "w" cap and physical KeyW).
  keyboard.press({ key: 'w', code: 'KeyZ' });
  keyboard.press({ key: 'z', code: 'KeyW' });
  assert.equal(keyboard.commands().filter(command => command === 'lookUp').length, 1, 'commands are deduplicated');
  keyboard.release({ key: 'w', code: 'KeyZ' });
  assert.ok(keyboard.isHeld('lookUp'), 'the other physical key is still down');
  keyboard.release({ key: 'z', code: 'KeyW' });
  assert.ok(!keyboard.isHeld('lookUp'));
});

test('releasing works even if Shift, CapsLock or a modifier changed the character', () => {
  const keyboard = controls.createKeyboardController();
  keyboard.press(keyEvent('a'));
  assert.equal(keyboard.release(keyEvent('A', { shiftKey: true })), 'lookLeft', 'Shift pressed mid-hold');
  assert.equal(keyboard.size, 0);
  keyboard.press(keyEvent('A'));
  assert.equal(keyboard.release(keyEvent('a')), 'lookLeft', 'CapsLock toggled mid-hold');
  keyboard.press(keyEvent('ArrowRight'));
  assert.equal(keyboard.release(keyEvent('ArrowRight', { ctrlKey: true })), 'right', 'release never consults modifiers');
  assert.equal(keyboard.size, 0);
  // Virtual keyboards: no code on either side, character case may differ.
  keyboard.press({ key: 'D', code: '' });
  assert.equal(keyboard.release({ key: 'd', code: '' }), 'lookRight');
  assert.equal(keyboard.size, 0);
});

test('a modifier chord clears held keys so nothing can stick behind ⌘', () => {
  const keyboard = controls.createKeyboardController();
  keyboard.press(keyEvent('ArrowUp'));
  keyboard.press(keyEvent('a'));
  assert.equal(keyboard.press(keyEvent('Meta')), null);
  assert.equal(keyboard.size, 0, 'pressing ⌘ while moving drops every held key');
  assert.equal(keyboard.press(keyEvent('ArrowLeft', { metaKey: true })), null, '⌘← is never tracked');
  assert.equal(keyboard.size, 0);
  // macOS swallows this keyup; the controller must not depend on it.
  assert.equal(keyboard.release(keyEvent('ArrowLeft')), null);
  keyboard.press(keyEvent('w'));
  keyboard.press(keyEvent('x', { ctrlKey: true }));
  assert.equal(keyboard.size, 0, 'any Ctrl chord clears as well');
  keyboard.press(keyEvent('w'));
  keyboard.press(keyEvent('Alt'));
  assert.equal(keyboard.size, 0, 'Alt on its own clears too');
  keyboard.press(keyEvent('w'));
  keyboard.press(keyEvent('Shift'));
  assert.equal(keyboard.size, 1, 'Shift is harmless and keeps the held key');
});

test('unrelated keys, composition and clear() behave predictably', () => {
  const keyboard = controls.createKeyboardController();
  keyboard.press(keyEvent('ArrowDown'));
  assert.equal(keyboard.press(keyEvent('r')), null, 'reset is not a movement command');
  assert.equal(keyboard.press(keyEvent('Escape')), null);
  assert.equal(keyboard.press(keyEvent('e')), null);
  assert.equal(keyboard.size, 1, 'unknown keys never clear held navigation keys');
  assert.equal(keyboard.press(keyEvent('w', { isComposing: true })), null);
  assert.equal(keyboard.size, 1, 'IME composition is ignored without side effects');
  assert.equal(keyboard.release(keyEvent('e')), null, 'releasing a key that was never held is harmless');
  assert.equal(keyboard.release(null), null);
  assert.equal(keyboard.press(null), null);
  assert.equal(keyboard.press(undefined), null);
  keyboard.clear();
  assert.equal(keyboard.size, 0);
  same(keyboard.commands(), []);
});

test('a fresh press is reported exactly once per physical press', () => {
  const keyboard = controls.createKeyboardController();
  let impulses = 0;
  const down = event => { const pressed = keyboard.press(event); if (pressed && pressed.fresh) impulses++; };
  down(keyEvent('ArrowUp'));
  for (let index = 0; index < 30; index++) down(keyEvent('ArrowUp', { repeat: true }));
  keyboard.release(keyEvent('ArrowUp'));
  down(keyEvent('ArrowUp'));
  keyboard.release(keyEvent('ArrowUp'));
  assert.equal(impulses, 2, 'two physical presses, thirty auto-repeats, two tap impulses');
});
