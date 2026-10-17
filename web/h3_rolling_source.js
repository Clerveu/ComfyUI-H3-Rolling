// Scrubbable player for H3 Rolling Source: watch the clip, set start/end from the playhead.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "H3Rolling.SourceScrubber",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3RollingSource") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      setupScrubber(this);
      return r;
    };
  },
});

function setupScrubber(node) {
  const W = (name) => node.widgets?.find((w) => w.name === name);

  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;height:100%;";

  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  video.style.cssText = "width:100%;flex:1 1 auto;min-height:0;background:#000;border-radius:4px;object-fit:contain;";

  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;align-items:center;font:12px sans-serif;color:#ccc;";
  const button = (label, fn) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "padding:2px 8px;cursor:pointer;border-radius:4px;border:1px solid #555;background:#2a2a2a;color:#ddd;";
    b.onpointerdown = (e) => e.stopPropagation();
    b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); fn(); };
    bar.appendChild(b);
  };
  const readout = document.createElement("div");
  readout.style.cssText = "font:12px monospace;color:#9ab;";

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "video/*";
  fileInput.style.display = "none";

  let playingRange = false;
  const round = (t) => Math.round(t * 100) / 100;

  button("⇤ Set start", () => setVal("start_seconds", round(video.currentTime)));
  button("Set end ⇥", () => setVal("end_seconds", round(video.currentTime)));
  button("▶ Play range", () => {
    video.currentTime = W("start_seconds")?.value ?? 0;
    playingRange = true;
    video.play();
  });
  button("Reset", () => { setVal("start_seconds", 0); setVal("end_seconds", 0); });
  button("Upload…", () => fileInput.click());

  root.append(video, bar, readout, fileInput);

  function setVal(name, v) {
    const w = W(name);
    if (!w) return;
    w.value = v;
    w.callback?.(v);
    app.graph.setDirtyCanvas(true, true);
    update();
  }

  function load() {
    const v = W("video")?.value;
    if (!v) return;
    const i = v.lastIndexOf("/");
    const filename = i >= 0 ? v.slice(i + 1) : v;
    const subfolder = i >= 0 ? v.slice(0, i) : "";
    const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
    if (video.dataset.src !== url) {
      video.dataset.src = url;
      video.src = url;
    }
  }

  function update() {
    const s = Number(W("start_seconds")?.value ?? 0);
    const e = Number(W("end_seconds")?.value ?? 0);
    const d = isFinite(video.duration) ? video.duration : 0;
    const end = e > 0 ? e : d;
    readout.textContent =
      `in ${s.toFixed(2)}s → out ${e > 0 ? e.toFixed(2) + "s" : "end"}  ` +
      `(${Math.max(0, end - s).toFixed(2)}s)   playhead ${video.currentTime.toFixed(2)}s`;
  }

  fileInput.onchange = async () => {
    const f = fileInput.files?.[0];
    fileInput.value = "";
    if (!f) return;
    const body = new FormData();
    body.append("image", f);
    body.append("type", "input");
    body.append("overwrite", "true");
    readout.textContent = `uploading ${f.name}…`;
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status !== 200) {
      readout.textContent = `upload failed: ${resp.status} ${resp.statusText}`;
      return;
    }
    const data = await resp.json();
    const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
    const vw = W("video");
    if (vw) {
      const values = vw.options?.values;
      if (Array.isArray(values) && !values.includes(name)) values.push(name);
      vw.value = name;
      vw.callback?.(name);
    }
    setVal("start_seconds", 0);
    setVal("end_seconds", 0);
    load();
  };

  video.addEventListener("timeupdate", () => {
    update();
    const e = Number(W("end_seconds")?.value ?? 0);
    if (playingRange && e > 0 && video.currentTime >= e) {
      video.pause();
      playingRange = false;
    }
  });
  video.addEventListener("loadedmetadata", update);
  video.addEventListener("seeked", update);

  node.addDOMWidget("scrubber", "h3_rolling_scrubber", root, {
    serialize: false,
    getMinHeight: () => 260,
  });

  const vw = W("video");
  if (vw) {
    const cb = vw.callback;
    vw.callback = function () {
      const r = cb?.apply(this, arguments);
      load();
      return r;
    };
  }
  for (const n of ["start_seconds", "end_seconds"]) {
    const w = W(n);
    if (!w) continue;
    const cb = w.callback;
    w.callback = function () {
      const r = cb?.apply(this, arguments);
      update();
      return r;
    };
  }
  const onConfigure = node.onConfigure;
  node.onConfigure = function () {
    const r = onConfigure?.apply(this, arguments);
    setTimeout(load, 0);
    return r;
  };
  setTimeout(load, 0);
  node.setSize([Math.max(node.size[0], 460), Math.max(node.size[1], 460)]);
}
