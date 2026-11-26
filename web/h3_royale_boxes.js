// The Royale node's text boxes (prompt, storyboard, arms) each get a drag handle at their bottom right corner.
// The node grows and shrinks with them, and each box's height is saved in the workflow with the node.
// For the Vue node renderer (Comfy.VueNodes.Enabled), where a node is page layout and sizes to what is in it.
import { app } from "../../scripts/app.js";

const TYPES = new Set(["H3RollingRoyale"]);
const KEY = "h3_box_heights";   // node.properties[KEY] = [prompt px, storyboard px, arms px]
const MIN = 64;

function nodeOf(el) {
  const holder = el.closest("[data-node-id]");
  if (!holder) return null;
  const node = app.graph?.getNodeById?.(Number(holder.dataset.nodeId)) ?? app.graph?.getNodeById?.(holder.dataset.nodeId);
  return node && TYPES.has(node.type) ? { node, holder } : null;
}

function dress(ta) {
  const hit = nodeOf(ta);
  if (!hit) return;
  const boxes = [...hit.holder.querySelectorAll("textarea")];
  const i = boxes.indexOf(ta);
  ta.style.resize = "vertical";
  ta.style.minHeight = MIN + "px";
  ta.style.maxHeight = "none";
  const saved = (hit.node.properties?.[KEY] || [])[i];
  if (saved && ta.dataset.h3Box !== "1") ta.style.height = saved + "px";
  ta.dataset.h3Box = "1";
}

function remember(ta) {
  const hit = nodeOf(ta);
  if (!hit || !ta.style.height) return;
  const boxes = [...hit.holder.querySelectorAll("textarea")];
  const heights = [...(hit.node.properties?.[KEY] || [])];
  const h = Math.round(parseFloat(ta.style.height));
  const i = boxes.indexOf(ta);
  if (!(h >= MIN) || heights[i] === h) return;
  heights[i] = h;
  hit.node.properties = hit.node.properties || {};
  hit.node.properties[KEY] = heights;
  app.graph?.setDirtyCanvas?.(true, true);
}

function sweep(root) {
  if (root.nodeType !== 1) return;
  if (root.tagName === "TEXTAREA") dress(root);
  else root.querySelectorAll?.("textarea").forEach(dress);
}

app.registerExtension({
  name: "H3Rolling.RoyaleBoxes",
  setup() {
    new MutationObserver((muts) => { for (const m of muts) m.addedNodes.forEach(sweep); })
      .observe(document.body, { childList: true, subtree: true });
    sweep(document.body);
    // a drag on a box's corner ends with the pointer coming up, wherever it is by then
    let held = null;
    document.addEventListener("pointerdown", (e) => { held = e.target?.tagName === "TEXTAREA" && e.target.dataset.h3Box ? e.target : null; }, true);
    document.addEventListener("pointerup", () => { if (held) remember(held); held = null; }, true);
  },
  // a workflow loaded into nodes that are already on screen: put its heights on them
  loadedGraphNode(node) {
    if (!TYPES.has(node.type)) return;
    requestAnimationFrame(() => {
      const holder = document.querySelector(`[data-node-id="${node.id}"]`);
      holder?.querySelectorAll("textarea").forEach((ta) => { delete ta.dataset.h3Box; dress(ta); });
    });
  },
});
