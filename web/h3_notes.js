// H3 Notes: a plain text box whose words are kept in a file in Comfy's user folder (h3_notes.txt), not in the
// workflow. Any workflow with the node in it shows the same notes, a workflow that was never saved loses nothing,
// and rebuilding a workflow (the Royale page's Load into workflow) can't overwrite them.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "H3Notes";
const FILE = "h3_notes.txt";
let text = null;          // what the file holds, once it has been read
let saveTimer = null;

async function read() {
  try {
    const r = await api.fetchApi(`/userdata/${encodeURIComponent(FILE)}`);
    text = r.ok ? await r.text() : "";
  } catch (e) { text = ""; }
  return text;
}

function save(value) {
  text = value;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    api.fetchApi(`/userdata/${encodeURIComponent(FILE)}?overwrite=true`, { method: "POST", headers: { "Content-Type": "text/plain" }, body: text })
      .catch(() => {});
  }, 500);
}

function fill(node) {
  const w = node.widgets?.find((x) => x.name === "notes");
  if (!w || text === null) return;
  if (w.value !== text) { w.value = text; w.callback?.(text); }
  const el = document.querySelector(`[data-node-id="${node.id}"] textarea`);
  if (el && el.value !== text) { el.value = text; el.dispatchEvent(new Event("input", { bubbles: true })); }
  app.graph?.setDirtyCanvas?.(true, true);
}

function notesNodes() { return (app.graph?._nodes || []).filter((n) => n.type === TYPE); }

app.registerExtension({
  name: "H3Rolling.Notes",
  async setup() {
    // typing in the box saves it, a moment after the last key
    document.addEventListener("input", (e) => {
      const t = e.target;
      if (t?.tagName !== "TEXTAREA" || !e.isTrusted) return;
      const holder = t.closest("[data-node-id]");
      const node = holder && (app.graph?.getNodeById?.(Number(holder.dataset.nodeId)) ?? app.graph?.getNodeById?.(holder.dataset.nodeId));
      if (node?.type === TYPE) save(t.value);
    }, true);
    window.addEventListener("pagehide", () => {
      if (saveTimer) { clearTimeout(saveTimer); navigator.sendBeacon?.(api.apiURL(`/userdata/${encodeURIComponent(FILE)}?overwrite=true`), new Blob([text], { type: "text/plain" })); }
    });
    await read();
    notesNodes().forEach(fill);
  },
  nodeCreated(node) {
    if (node.type !== TYPE) return;
    const put = () => setTimeout(() => fill(node), 60);
    if (text === null) read().then(put); else put();
  },
  // a loaded workflow carries whatever the box held when it was made; the file wins
  loadedGraphNode(node) {
    if (node.type !== TYPE) return;
    read().then(() => setTimeout(() => fill(node), 60));
  },
});
