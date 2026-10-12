/* DOM, decoded image buffers, and pointer/keyboard controls; no geometry. */
(async function () {
  "use strict";
  const payload = JSON.parse(document.getElementById("reveal-payload").textContent);
  const root = document.getElementById("reveal-root"), canvas = document.getElementById("reveal-canvas");
  const context = canvas.getContext("2d"), status = document.getElementById("status");
  const buttons = Object.fromEntries(["start", "pause", "preview", "reset"].map(name => [name, document.getElementById(name)]));
  const brush = document.getElementById("brush-size"), cursor = document.getElementById("brush-cursor");
  canvas.width = payload.width; canvas.height = payload.height;
  const state = new RevealState(payload.width, payload.height);
  let layers, frame = context.createImageData(payload.width, payload.height), pending = false, saveTimer;
  let storageAvailable = true;
  function persist() {
    clearTimeout(saveTimer);
    try {
      let binary = "";
      for (let offset = 0; offset < state.depth.length; offset += 8192) binary += String.fromCharCode(...state.depth.subarray(offset, offset + 8192));
      sessionStorage.setItem(payload.storageKey, JSON.stringify({depth: btoa(binary), brush: brush.value}));
    } catch (_) { storageAvailable = false; updateStatus(); }
  }
  function scheduleSave() { clearTimeout(saveTimer); saveTimer = setTimeout(persist, 250); }
  function updateStatus() {
    if (!state.active || state.previewFinal) cursor.style.display = "none";
    status.textContent = state.previewFinal ? "Previewing Final · history preserved" : state.active ? "Reveal Active" : "Reveal Paused";
    if (!storageAvailable) status.textContent += " · session save unavailable";
    buttons.preview.textContent = state.previewFinal ? "Return to Reveal" : "Preview Final";
    buttons.preview.setAttribute("aria-pressed", String(state.previewFinal));
    buttons.start.setAttribute("aria-pressed", String(state.active));
    root.dataset.active = String(state.active); root.dataset.preview = String(state.previewFinal);
  }
  function paint() {
    if (pending || !layers) return;
    pending = true;
    requestAnimationFrame(() => {
      state.composite(layers, frame.data); context.putImageData(frame, 0, 0);
      pending = false; updateStatus();
    });
  }
  try {
    layers = await Promise.all(payload.layers.map(layer => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        if (image.width !== payload.width || image.height !== payload.height) { reject(new Error("Layer size mismatch")); return; }
        const offscreen = document.createElement("canvas"); offscreen.width = payload.width; offscreen.height = payload.height;
        const ctx = offscreen.getContext("2d", {willReadFrequently: true}); ctx.drawImage(image, 0, 0);
        resolve(ctx.getImageData(0, 0, payload.width, payload.height).data);
      };
      image.onerror = () => reject(new Error("Could not load image layers")); image.src = layer.src;
    })));
    try {
      const saved = JSON.parse(sessionStorage.getItem(payload.storageKey) || "null");
      if (saved) {
        const bytes = Uint8Array.from(atob(saved.depth), char => char.charCodeAt(0)); state.restore(bytes);
        if (Number(saved.brush) >= 2 && Number(saved.brush) <= 15) brush.value = saved.brush;
      }
    } catch (_) { storageAvailable = false; }
    for (const button of Object.values(buttons)) button.disabled = false;
    root.dataset.ready = "true"; root.dataset.layerCount = String(layers.length);
    buttons.start.addEventListener("click", () => { state.resume(); paint(); });
    buttons.pause.addEventListener("click", () => { state.pause(); paint(); persist(); });
    buttons.preview.addEventListener("click", () => { state.togglePreview(); paint(); });
    buttons.reset.addEventListener("click", () => { state.reset(); paint(); persist(); });
    root.addEventListener("keydown", event => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
      if (event.key.toLowerCase() === "s") { event.preventDefault(); state.resume(); paint(); }
      if (event.key.toLowerCase() === "q") { event.preventDefault(); state.pause(); paint(); persist(); }
    });
    brush.addEventListener("input", () => { state.endPass(); scheduleSave(); });
    canvas.addEventListener("pointerdown", () => canvas.focus({preventScroll: true}));
    canvas.addEventListener("pointermove", event => {
      const rect = canvas.getBoundingClientRect();
      const displayRadius = rect.width * Number(brush.value) / 100;
      cursor.style.width = cursor.style.height = `${displayRadius * 2}px`;
      cursor.style.left = `${event.clientX - rect.left - displayRadius}px`;
      cursor.style.top = `${event.clientY - rect.top - displayRadius}px`;
      cursor.style.display = state.previewFinal || !state.active ? "none" : "block";
      const changed = state.revealPath((event.clientX - rect.left) * payload.width / rect.width,
        (event.clientY - rect.top) * payload.height / rect.height, payload.width * Number(brush.value) / 100);
      if (changed) { paint(); scheduleSave(); }
    });
    canvas.addEventListener("pointerleave", () => { state.endPass(); cursor.style.display = "none"; persist(); });
    canvas.addEventListener("pointerup", event => {
      if (event.pointerType !== "mouse") { state.endPass(); cursor.style.display = "none"; persist(); }
    });
    window.addEventListener("pagehide", persist);
    // Read-only diagnostics make browser validation possible without exposing mutation hooks.
    Object.defineProperty(window, "mathLensRevealDiagnostics", {value: () => ({
      ready: true, layers: payload.layers.map(layer => layer.name), active: state.active,
      previewFinal: state.previewFinal, depth: Array.from(state.depth), storageKey: payload.storageKey,
    })});
    paint();
  } catch (error) {
    status.textContent = error.message; status.classList.add("error"); root.dataset.ready = "error";
  }
})();
