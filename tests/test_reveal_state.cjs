const test = require("node:test");
const assert = require("node:assert/strict");
const {RevealState} = require("../src/math_lens/ui/assets/reveal_state.js");

test("paused movement retains Layer 5; resume enables a local reveal", () => {
  const state = new RevealState(40, 30);
  state.revealCircle(20, 15, 4);
  assert.equal(state.depth[15 * 40 + 20], 5);
  state.resume(); state.revealCircle(20, 15, 4);
  assert.equal(state.depth[15 * 40 + 20], 4);
  assert.equal(state.depth[0], 5);
});
test("overlap is one sweep; leaving and returning descends through all layers", () => {
  const state = new RevealState(40, 30); state.resume();
  state.revealCircle(20, 15, 4); state.revealCircle(21, 15, 4);
  assert.equal(state.depth[15 * 40 + 20], 4);
  for (let pass = 0; pass < 6; pass++) { state.endPass(); state.revealCircle(20, 15, 4); }
  assert.equal(state.depth[15 * 40 + 20], 0);
  assert.equal(state.depth[0], 5);
});
test("pause and resume retain the depth map", () => {
  const state = new RevealState(40, 30); state.resume(); state.revealCircle(20, 15, 4);
  const saved = Array.from(state.depth); state.pause(); state.revealCircle(0, 0, 4);
  assert.deepEqual(Array.from(state.depth), saved); state.resume();
  assert.deepEqual(Array.from(state.depth), saved);
});
test("Preview Final changes compositing without editing reveal history", () => {
  const state = new RevealState(4, 4); state.resume(); state.revealCircle(2, 2, 2);
  const saved = Array.from(state.depth);
  const layers = Array.from({length: 6}, (_, layer) => new Uint8ClampedArray(4 * 4 * 4).fill(layer * 30));
  const before = state.composite(layers);
  state.togglePreview(); state.revealCircle(1, 1, 4);
  assert.equal(state.composite(layers)[0], 150);
  assert.deepEqual(Array.from(state.depth), saved);
  state.togglePreview(); assert.deepEqual(state.composite(layers), before);
});
test("Reset restores Layer 5 and history can be restored across component reloads", () => {
  const state = new RevealState(40, 30); state.resume(); state.revealCircle(20, 15, 4);
  const restored = new RevealState(40, 30); restored.restore(state.depth);
  assert.deepEqual(restored.depth, state.depth); state.reset();
  assert.ok(state.depth.every(depth => depth === 5)); assert.equal(state.active, false);
});
test("fast pointer paths do not leave gaps or skip layers", () => {
  const state = new RevealState(80, 30); state.resume(); state.revealCircle(10, 15, 4);
  state.revealPath(65, 15, 4);
  for (let x = 10; x <= 65; x++) assert.equal(state.depth[15 * 80 + x], 4);
});
test("sidebar commands are applied once, even when Streamlit repeats render args", () => {
  const state = new RevealState(40, 30);
  assert.ok(state.applyCommand({id:"start-1",action:"resume"}));
  state.revealCircle(20,15,4);
  state.applyCommand({id:"preview-1",action:"preview"});
  assert.equal(state.previewFinal,true);
  assert.equal(state.applyCommand({id:"preview-1",action:"preview"}),false);
  assert.equal(state.previewFinal,true);
  state.applyCommand({id:"preview-2",action:"preview"});
  assert.equal(state.previewFinal,false);
  assert.equal(state.depth[15*40+20],4);
  state.applyCommand({id:"reset-1",action:"reset"});
  assert.ok(state.depth.every(value=>value===5));
  assert.equal(state.active,false);
});
test("restored command IDs prevent a preview toggle replay after remount", () => {
  const original = new RevealState(40,30);
  const command = {id:"saved-preview",action:"preview"};
  original.applyCommand(command);
  const restored = new RevealState(40,30);
  restored.restore(original.depth);
  restored.lastCommand = original.lastCommand;
  restored.previewFinal = original.previewFinal;
  assert.equal(restored.applyCommand(command),false);
  assert.equal(restored.previewFinal,true);
});
