/* Streamlit canvas transport. Sidebar commands are discrete; brushing stays local. */
(function () {
  "use strict";
  const root = document.getElementById("reveal-root"), canvas = document.getElementById("reveal-canvas");
  const context = canvas.getContext("2d"), cursor = document.getElementById("brush-cursor");
  let payload, state, layers, frame, brush = 5, pending = false, saveTimer, lastStatus, lastHeight;
  let storageAvailable = true, queue = Promise.resolve();
  const send = (type, data) => window.parent.postMessage({isStreamlitMessage:true, type, ...data}, "*");
  function publish() {
    const value = {storageKey:payload.storageKey, ready:true, active:state.active,
      previewFinal:state.previewFinal, storageAvailable};
    const serialized = JSON.stringify(value);
    if (lastStatus !== serialized) {lastStatus = serialized; send("streamlit:setComponentValue", {value, dataType:"json"});}
    root.dataset.active = String(state.active); root.dataset.preview = String(state.previewFinal);
    if (!state.active || state.previewFinal) cursor.style.display = "none";
  }
  function persist() {
    clearTimeout(saveTimer);
    if (!state) return;
    try {
      let binary = "";
      for (let offset=0; offset<state.depth.length; offset+=8192) binary += String.fromCharCode(...state.depth.subarray(offset,offset+8192));
      sessionStorage.setItem(payload.storageKey, JSON.stringify({depth:btoa(binary), active:state.active,
        preview:state.previewFinal, lastCommand:state.lastCommand}));
    } catch (_) {storageAvailable = false; publish();}
  }
  function paint() {
    if (pending || !layers) return;
    pending = true;
    const paintState=state, paintLayers=layers, paintFrame=frame;
    requestAnimationFrame(() => {
      pending=false;
      if(state!==paintState){paint();return;}
      paintState.composite(paintLayers,paintFrame.data);context.putImageData(paintFrame,0,0);
    });
  }
  function resize() {
    if (!payload) return;
    let viewportHeight = window.screen.availHeight || 900;
    try {viewportHeight = window.parent.innerHeight;} catch (_) { /* isolated host fallback */ }
    // Match the static result's viewport budget, including title and margins.
    const maximumHeight = Math.max(180,Math.min(780,viewportHeight-280));
    root.style.maxWidth = `${maximumHeight*payload.width/payload.height}px`;
    const height = Math.ceil(root.getBoundingClientRect().height)+2;
    if (height !== lastHeight) {lastHeight=height; send("streamlit:setFrameHeight",{height});}
  }
  async function render(args) {
    const next = args.payload;
    if (!payload || next.storageKey !== payload.storageKey) {
      persist(); payload=next; layers=null; state=new RevealState(payload.width,payload.height);
      storageAvailable=true; lastStatus=null;
      canvas.width=payload.width; canvas.height=payload.height;
      frame=context.createImageData(payload.width,payload.height);
      layers=await Promise.all(payload.layers.map(layer => new Promise((resolve,reject) => {
        const image=new Image();
        image.onload=() => {
          if(image.width!==payload.width || image.height!==payload.height){reject(new Error("Layer size mismatch"));return;}
          const offscreen=document.createElement("canvas");offscreen.width=payload.width;offscreen.height=payload.height;
          const ctx=offscreen.getContext("2d",{willReadFrequently:true});ctx.drawImage(image,0,0);
          resolve(ctx.getImageData(0,0,payload.width,payload.height).data);
        };
        image.onerror=()=>reject(new Error("Could not load image layers"));image.src=layer.src;
      })));
      try {
        const saved=JSON.parse(sessionStorage.getItem(payload.storageKey)||"null");
        if(saved){state.restore(Uint8Array.from(atob(saved.depth),char=>char.charCodeAt(0)));
          state.active=Boolean(saved.active);state.previewFinal=Boolean(saved.preview);state.lastCommand=saved.lastCommand||null;}
      } catch (_) {storageAvailable=false;}
      root.dataset.ready="true";
    }
    const nextBrush=Math.max(2,Math.min(15,Number(args.brush)||5));
    if(brush!==nextBrush){brush=nextBrush;state.endPass();}
    const command=args.command;
    if((!command?.storageKey || command.storageKey===payload.storageKey) && state.applyCommand(command))persist();
    resize();paint();publish();
  }
  window.addEventListener("message",event => {
    if(event.source!==window.parent || event.data?.type!=="streamlit:render")return;
    queue=queue.then(()=>render(event.data.args)).catch(error=>{
      root.dataset.ready="error";document.getElementById("error").textContent=error.message;
      send("streamlit:setFrameHeight",{height:80});
    });
  });
  canvas.addEventListener("pointerdown",()=>canvas.focus({preventScroll:true}));
  canvas.addEventListener("keydown",event => {
    if(!state || event.ctrlKey || event.metaKey || event.altKey || event.repeat)return;
    const key=event.key.toLowerCase();
    if(key!=="s" && key!=="q")return;
    event.preventDefault();if(key==="s")state.resume();else state.pause();
    persist();paint();publish();
  });
  canvas.addEventListener("pointermove",event => {
    if(!state || !layers)return;
    const rect=canvas.getBoundingClientRect(), radius=rect.width*brush/100;
    cursor.style.width=cursor.style.height=`${radius*2}px`;
    cursor.style.left=`${event.clientX-rect.left-radius}px`;cursor.style.top=`${event.clientY-rect.top-radius}px`;
    cursor.style.display=state.active && !state.previewFinal?"block":"none";
    if(state.revealPath((event.clientX-rect.left)*payload.width/rect.width,
      (event.clientY-rect.top)*payload.height/rect.height,payload.width*brush/100)){
      paint();clearTimeout(saveTimer);saveTimer=setTimeout(persist,250);
    }
  });
  function endPass(){if(state)state.endPass();cursor.style.display="none";persist();}
  canvas.addEventListener("pointerleave",endPass);
  canvas.addEventListener("pointerup",event=>{if(event.pointerType!=="mouse")endPass();});
  // The iframe's fixed height may not change when only the app viewport height
  // changes. Listen to the same-origin host's resize as well as our own.
  let hostWindow;
  try {hostWindow=window.parent;hostWindow.addEventListener("resize",resize);} catch (_) {hostWindow=null;}
  window.addEventListener("pagehide",()=>{persist();if(hostWindow)hostWindow.removeEventListener("resize",resize);});
  window.addEventListener("resize",resize);new ResizeObserver(resize).observe(root);
  Object.defineProperty(window,"mathLensRevealDiagnostics",{value:()=>state && layers?{
    ready:true,layers:payload.layers.map(layer=>layer.name),active:state.active,previewFinal:state.previewFinal,
    depth:Array.from(state.depth),storageKey:payload.storageKey,brush,
  }: {ready:false}});
  send("streamlit:componentReady",{apiVersion:1});
})();
