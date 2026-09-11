/* Pure camera math and keyboard-navigation state for Vestra Studio; independently testable. */
(function (root) {
  'use strict';
  function normalize(vector) { const length = Math.hypot(...vector) || 1; return vector.map(value => value / length); }
  function cross(left, right) { return [left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0]]; }
  function quaternionMultiply(left, right) { return [left[3] * right[0] + left[0] * right[3] + left[1] * right[2] - left[2] * right[1], left[3] * right[1] - left[0] * right[2] + left[1] * right[3] + left[2] * right[0], left[3] * right[2] + left[0] * right[1] - left[1] * right[0] + left[2] * right[3], left[3] * right[3] - left[0] * right[0] - left[1] * right[1] - left[2] * right[2]]; }
  function normalizeQuaternion(quaternion) { const length = Math.hypot(...quaternion); return Number.isFinite(length) && length > 1e-12 ? quaternion.map(value => value / length) : [0, 0, 0, 1]; }
  function axisAngle(axis, angle) { const sine = Math.sin(angle / 2), length = Math.hypot(...axis) || 1; return [axis[0] / length * sine, axis[1] / length * sine, axis[2] / length * sine, Math.cos(angle / 2)]; }
  function rotate(quaternion, vector) { const unit = normalizeQuaternion(quaternion); return quaternionMultiply(quaternionMultiply(unit, [vector[0], vector[1], vector[2], 0]), [-unit[0], -unit[1], -unit[2], unit[3]]).slice(0, 3); }
  function orbit(orientation, horizontal, vertical) { let next = normalizeQuaternion(quaternionMultiply(axisAngle([0, 1, 0], horizontal), orientation)); const right = rotate(next, [1, 0, 0]); return normalizeQuaternion(quaternionMultiply(axisAngle(right, vertical), next)); }
  // Sign conventions for the first-person look controller. They follow the
  // right-hand rule in the viewer frame (+X right, +Y up, camera at rest looks
  // down -Z), and the tests pin them to the rendered forward vector so a
  // keyboard label can never drift from what the screen does:
  //   yaw   > 0  turns LEFT  (counter-clockwise when seen from above)
  //   pitch > 0  looks UP
  // Keyboard navigation deliberately has no roll. Yaw is wrapped to keep the
  // stored angle small; pitch is clamped before the vertical singularity, and
  // the *stored* pitch must be clamped too — otherwise holding "look up" would
  // bank invisible excess that "look down" has to unwind before anything moves.
  const MAX_PITCH = Math.PI / 2 - .02;
  function clampPitch(pitch) { return Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch)); }
  function wrapYaw(yaw) { const turn = 2 * Math.PI; return ((yaw + Math.PI) % turn + turn) % turn - Math.PI; }
  function lookOrientation(yaw, pitch) {
    return normalizeQuaternion(quaternionMultiply(axisAngle([0, 1, 0], yaw), axisAngle([1, 0, 0], clampPitch(pitch))));
  }
  // Inverse of lookOrientation for a forward direction (roll is discarded):
  // lets a pose adopted from calibrated camera evidence continue seamlessly
  // under keyboard control instead of snapping back to stale angles.
  function lookAnglesFromForward(forward) {
    const f = normalize(forward);
    return { yaw: wrapYaw(Math.atan2(-f[0], -f[2])), pitch: clampPitch(Math.asin(Math.max(-1, Math.min(1, f[1])))) };
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
  // Movement is relative to the current view; vertical movement stays on the
  // world axis so looking up never lifts the camera off its height.
  function movementDirection(basis, command) {
    switch (command) {
      case 'forward': return basis.forward;
      case 'backward': return basis.backward;
      case 'left': return basis.right.map(value => -value);
      case 'right': return basis.right;
      case 'up': return [0, 1, 0];
      case 'down': return [0, -1, 0];
      default: return null;
    }
  }
  // One discrete nudge by a ratio of the world scale. There is intentionally
  // no minimum step: a floor would make small worlds move per frame rather
  // than per second. Continuous motion goes through stepFirstPerson.
  function move(center, orientation, distance, command, ratio = .06) { const direction = movementDirection(cameraBasis(orientation), command); return direction ? center.map((value, axis) => value + direction[axis] * distance * ratio) : center.slice(); }
  // Look commands as (yaw, pitch) signs; see the convention block above.
  const LOOK_SIGNS = { lookLeft: [1, 0], lookRight: [-1, 0], lookUp: [0, 1], lookDown: [0, -1] };
  // Key resolution prefers the physical key (`KeyboardEvent.code`): it is
  // layout independent and, unlike `key`, identical on keydown and keyup even
  // when Shift or CapsLock changes the produced character mid-press. The
  // printed character is the fallback for virtual keyboards without codes and
  // for layouts whose W/A/S/D caps sit elsewhere, so both conventions work.
  const COMMANDS_BY_CODE = { ArrowUp: 'forward', ArrowDown: 'backward', ArrowLeft: 'left', ArrowRight: 'right', KeyW: 'lookUp', KeyS: 'lookDown', KeyA: 'lookLeft', KeyD: 'lookRight' };
  const COMMANDS_BY_KEY = { ArrowUp: 'forward', ArrowDown: 'backward', ArrowLeft: 'left', ArrowRight: 'right', w: 'lookUp', s: 'lookDown', a: 'lookLeft', d: 'lookRight' };
  const MODIFIER_KEYS = new Set(['Meta', 'Control', 'Alt', 'AltGraph', 'OS', 'Hyper', 'Super']);
  function commandForCode(code) { return (typeof code === 'string' && COMMANDS_BY_CODE[code]) || null; }
  function commandForKey(key) { if (typeof key !== 'string') return null; return COMMANDS_BY_KEY[key.length === 1 ? key.toLowerCase() : key] || null; }
  // A chord with Meta/Ctrl/Alt is never navigation: those combinations belong
  // to the browser (⌘← is history back) and, on macOS, the OS swallows the
  // keyup of any key released while ⌘ is held, which would leave it stuck.
  function isModifierChord(event) { return Boolean(event && (event.metaKey || event.ctrlKey || event.altKey || MODIFIER_KEYS.has(event.key))); }
  function commandForEvent(event) {
    if (!event || event.isComposing || isModifierChord(event)) return null;
    return commandForCode(event.code) || commandForKey(event.key);
  }
  // The identity a held key is tracked under. `code` is stable across the
  // whole press; the character is only used when no physical code exists.
  function keyIdentity(event) { const code = event && event.code; if (typeof code === 'string' && code && code !== 'Unidentified') return code; const key = event && typeof event.key === 'string' ? event.key : ''; return 'key:' + (key.length === 1 ? key.toLowerCase() : key); }
  // Tracks which navigation keys are physically held. Invariants:
  //   * a key is held from its first keydown until its keyup, regardless of
  //     how many auto-repeat keydowns the OS delivers in between;
  //   * `press` reports `fresh` exactly once per physical press, so a tap
  //     impulse is never re-applied by auto-repeat;
  //   * a modifier chord clears everything (see isModifierChord);
  //   * `release` never consults modifiers or the command table, so a key can
  //     always be let go even if its meaning changed while it was down.
  function createKeyboardController() {
    const held = new Map();
    return {
      press(event) {
        if (!event || event.isComposing) return null;
        if (isModifierChord(event)) { held.clear(); return null; }
        const command = commandForEvent(event);
        if (!command) return null;
        const id = keyIdentity(event), fresh = !held.has(id);
        held.set(id, command);
        return { command, fresh };
      },
      release(event) {
        if (!event) return null;
        const id = keyIdentity(event), command = held.get(id) || null;
        held.delete(id);
        return command;
      },
      clear() { held.clear(); },
      commands() { return [...new Set(held.values())]; },
      isHeld(command) { for (const value of held.values()) if (value === command) return true; return false; },
      get size() { return held.size; },
    };
  }
  // Advances a first-person camera by `seconds` under the given held commands.
  // Pure: returns a new state and never mutates its input. Speeds are per
  // second, so the result depends only on elapsed time, never on frame rate.
  // Simultaneous movement keys are combined and normalized, so a diagonal is
  // not faster than a straight line and opposite keys cancel exactly.
  function stepFirstPerson(state, commands, seconds, rates) {
    const dt = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
    const turnRate = Number.isFinite(rates && rates.turn) ? rates.turn : 0, moveRate = Number.isFinite(rates && rates.move) ? rates.move : 0;
    let yaw = state.yaw, pitch = state.pitch, turned = false;
    for (const command of commands) {
      const signs = LOOK_SIGNS[command];
      if (!signs) continue;
      yaw += signs[0] * turnRate * dt;
      pitch += signs[1] * turnRate * dt;
      turned = true;
    }
    yaw = wrapYaw(yaw);
    pitch = clampPitch(pitch);
    const orientation = lookOrientation(yaw, pitch), basis = cameraBasis(orientation), direction = [0, 0, 0];
    for (const command of commands) {
      const axis = movementDirection(basis, command);
      if (axis) for (let index = 0; index < 3; index++) direction[index] += axis[index];
    }
    const length = Math.hypot(...direction), moved = length > 1e-9, scale = moved ? moveRate * dt / Math.max(length, 1) : 0;
    const eye = state.eye.map((value, axis) => value + direction[axis] * scale);
    return { eye, yaw, pitch, orientation, moved, turned };
  }
  // DA3 camera coordinates are +X right, +Y down, +Z forward. WebGL's scene
  // presentation uses +X right, +Y up, -Z forward. This proper 180° X rotation
  // keeps positions, normals, and camera rays coherent.
  function cameraToViewer(vector) { return [vector[0], -vector[1], -vector[2]]; }
  root.VestraCameraControls = { MAX_PITCH, axisAngle, cameraBasis, cameraToViewer, clampPitch, commandForCode, commandForEvent, commandForKey, createKeyboardController, isModifierChord, keyIdentity, lookAnglesFromForward, lookOrientation, move, movementDirection, normalizeQuaternion, orbit, orientationFromBasis, quaternionMultiply, rotate, stepFirstPerson, viewMatrix, wrapYaw };
}(typeof window === 'undefined' ? globalThis : window));
