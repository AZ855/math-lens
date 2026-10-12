/* Pure per-pixel reveal state; shared by the canvas and targeted Node checks. */
(function (root) {
  "use strict";
  class RevealState {
    constructor(width, height, layerCount = 6) {
      if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || layerCount !== 6) {
        throw new Error("RevealState requires positive image dimensions and six layers");
      }
      this.width = width;
      this.height = height;
      this.top = layerCount - 1;
      this.depth = new Uint8Array(width * height).fill(this.top);
      this.active = false;
      this.previewFinal = false;
      this.previous = null;
      this.lastCommand = null;
    }
    resume() { this.active = true; this.previous = null; }
    pause() { this.active = false; this.previous = null; }
    endPass() { this.previous = null; }
    togglePreview() { this.previewFinal = !this.previewFinal; this.previous = null; }
    reset() { this.depth.fill(this.top); this.previewFinal = false; this.pause(); }
    applyCommand(command) {
      if (!command || !command.id || command.id === this.lastCommand) return false;
      switch (command.action) {
        case "resume": this.resume(); break;
        case "pause": this.pause(); break;
        case "preview": this.togglePreview(); break;
        case "reset": this.reset(); break;
        default: throw new Error("Unknown reveal command");
      }
      this.lastCommand = command.id;
      return true;
    }
    restore(values) {
      if (values.length !== this.depth.length || values.some(value => !Number.isInteger(value) || value < 0 || value > this.top)) {
        throw new Error("Invalid reveal history");
      }
      this.depth.set(values);
      this.previous = null;
    }
    revealCircle(x, y, radius) {
      if (!this.active || this.previewFinal || ![x, y, radius].every(Number.isFinite) || radius <= 0) return 0;
      const left = Math.max(0, Math.floor(x - radius)), right = Math.min(this.width - 1, Math.ceil(x + radius));
      const top = Math.max(0, Math.floor(y - radius)), bottom = Math.min(this.height - 1, Math.ceil(y + radius));
      const previous = this.previous;
      let changed = 0;
      for (let py = top; py <= bottom; py++) {
        for (let px = left; px <= right; px++) {
          if ((px - x) ** 2 + (py - y) ** 2 > radius ** 2) continue;
          // Overlapping mouse events form one sweep, not five instant passes.
          // A pixel can descend again after the brush leaves and returns.
          if (previous && (px - previous.x) ** 2 + (py - previous.y) ** 2 <= previous.radius ** 2) continue;
          const index = py * this.width + px;
          if (this.depth[index] > 0) { this.depth[index]--; changed++; }
        }
      }
      this.previous = {x, y, radius};
      return changed;
    }
    revealPath(x, y, radius) {
      if (!this.active || this.previewFinal) return 0;
      const previous = this.previous;
      if (!previous) return this.revealCircle(x, y, radius);
      const distance = Math.hypot(x - previous.x, y - previous.y);
      const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius / 3)));
      let changed = 0;
      for (let step = 1; step <= steps; step++) {
        changed += this.revealCircle(previous.x + (x - previous.x) * step / steps,
          previous.y + (y - previous.y) * step / steps, radius);
      }
      return changed;
    }
    composite(layers, destination) {
      if (layers.length !== 6 || layers.some(layer => layer.length !== this.depth.length * 4)) throw new Error("Invalid image layers");
      const output = destination || new Uint8ClampedArray(this.depth.length * 4);
      for (let index = 0; index < this.depth.length; index++) {
        const offset = index * 4, source = layers[this.previewFinal ? this.top : this.depth[index]];
        output[offset] = source[offset]; output[offset + 1] = source[offset + 1];
        output[offset + 2] = source[offset + 2]; output[offset + 3] = 255;
      }
      return output;
    }
  }
  root.RevealState = RevealState;
  if (typeof module !== "undefined" && module.exports) module.exports = {RevealState};
})(typeof globalThis !== "undefined" ? globalThis : window);
