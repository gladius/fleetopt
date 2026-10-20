// Builds presentation/fleetopt-demo.pptx:  node build.js
// A theme, four layouts and two slides to start from. Add slides at the bottom; restyle in THEME.
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR
  ? path.join(process.env.PPTX_SKILL_DIR, "scripts/apply_theme.js")
  : path.join(require("os").homedir(),
      ".claude/skills/synced/7dca438e-7544-4f3a-b15a-a3a68dceae69_e5793525-489b-4d90-9efe-8254ff91a809/pptx/scripts/apply_theme.js"));

// Black, white and one red, with greys between. Text is near-black on white, not pure black;
// red is used once a slide, small, for the one thing that matters on it.
const THEME = {
  name: "fleetopt",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "111111", lt1: "FFFFFF", dk2: "333333", lt2: "F6F6F6",
    accent1: "EE0000", accent2: "747676", accent3: "D8DADA", accent4: "A7A7A7",
    accent5: "000000", accent6: "B00000", hlink: "333333", folHlink: "747676",
  },
};
const OUT = path.join(__dirname, "fleetopt-demo.pptx");

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "fleetopt";
  pres.author = "Central AI team";
  const C = pres.SchemeColor;
  const W = 13.333;

  // ---------------------------------------------------------------- layouts
  pres.defineSlideMaster({
    title: "TITLE", background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 10.5, h: 1.3, fontSize: 60, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.35, w: 11.6, h: 1.1, fontSize: 22, color: C.accent3, align: "left", valign: "top", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "meta", type: "body", x: 0.8, y: 6.65, w: 9, h: 0.35, fontSize: 12, color: C.accent4, align: "left", valign: "middle", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "SECTION", background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.7, w: 11.5, h: 1.2, fontSize: 44, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 4.05, w: 10, h: 0.9, fontSize: 20, color: C.accent3, align: "left", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  const frame = [
    { text: { text: "fleetopt", options: { x: 0.6, y: 6.98, w: 3, h: 0.3, fontSize: 10, color: C.accent2, align: "left", valign: "middle", margin: 0 } } },
  ];
  const number = { x: W - 1.2, y: 6.98, w: 0.6, h: 0.3, fontSize: 10, color: C.accent2, align: "right" };
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 }, slideNumber: number,
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
      ...frame,
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT_TEXT", background: { color: C.background1 }, slideNumber: number,
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.6, w: 12.1, h: 5.0, fontSize: 18, color: C.text2, align: "left", valign: "top", margin: 0 }, text: "" } },
      ...frame,
    ],
  });

  // ---------------------------------------------------------------- helpers
  const text = (slide, value, o) => slide.addText(value, { isTextBox: true, margin: 0, valign: "top", align: "left", ...o });
  const card = (slide, name, o) => slide.addShape(pres.ShapeType.roundRect, {
    objectName: name, rectRadius: 0.08, line: { type: "none" }, fill: { color: C.background2 }, ...o });
  const label = (slide, value, x, y, w) => text(slide, value, {
    objectName: `label ${value}`, x, y, w, h: 0.3, fontSize: 11, bold: true, color: C.accent2, charSpacing: 1.5 });
  const arrow = (slide, name, x, y, w, both) => slide.addShape(pres.ShapeType.line, {
    objectName: name, x, y, w, h: 0, line: { color: C.accent2, width: 1.5, endArrowType: "triangle", beginArrowType: both ? "triangle" : "none" } });

  // ---------------------------------------------------------------- 1. title
  pres.addSection({ title: "Opening" });
  {
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Opening" });
    s.addText("fleetopt", { placeholder: "title" });
    s.addText("Experts that review and optimise AI agents, and prove every change", { placeholder: "subtitle" });
    s.addText("Central AI team  ·  October 2026", { placeholder: "meta" });
    // What it does, in five steps; the last one is the point.
    const steps = ["Start", "Measure", "Find", "Change", "Prove"];
    const w = 1.55, gap = 0.45, y = 5.05, h = 0.5;
    steps.forEach((step, i) => {
      const x = 0.8 + i * (w + gap), last = i === steps.length - 1;
      s.addShape(pres.ShapeType.roundRect, {
        objectName: `step ${step}`, x, y, w, h, rectRadius: 0.25,
        fill: last ? { color: C.accent1 } : { color: C.text1 },
        line: last ? { type: "none" } : { color: C.accent2, width: 1 } });
      text(s, step, { objectName: `step ${step} text`, x, y, w, h, fontSize: 14, bold: last, align: "center", valign: "middle",
                      color: last ? C.background1 : C.accent3 });
      if (!last) arrow(s, `step ${step} to next`, x + w + 0.07, y + h / 2, gap - 0.14);
    });
    s.addNotes("fleetopt sends an expert to a team's AI agent. The expert starts the agent in the team's own setup, measures it, "
      + "finds what its expertise looks for, changes the code on a branch, and keeps a change only when it is proven. "
      + "The last step is the point: it refuses what it cannot prove.");
  }

  // ---------------------------------------------------------------- 2. architecture
  pres.addSection({ title: "How it works" });
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
    s.addText("Architecture: experts are prose, proof is code", { placeholder: "title" });

    // Left: the catalogue of experts
    const ax = 0.6, aw = 3.05;
    label(s, "CENTRAL CATALOGUE", ax, 1.55, aw);
    [["cost", "Where tokens and money are wasted"],
     ["design", "Whether the design fits the job"],
     ["tester", "Requests that would check the agent"]].forEach(([name, does], i) => {
      const y = 1.95 + i * 1.3;
      card(s, `expert ${name}`, { x: ax, y, w: aw, h: 1.12 });
      text(s, name, { objectName: `expert ${name} name`, x: ax + 0.22, y: y + 0.14, w: aw - 0.44, h: 0.34, fontSize: 18, bold: true, color: C.text1 });
      text(s, does, { objectName: `expert ${name} does`, x: ax + 0.22, y: y + 0.5, w: aw - 0.44, h: 0.55, fontSize: 14, color: C.text2 });
    });
    text(s, "Markdown only: a guide, checks and skills. Pulled on demand.", {
      objectName: "catalogue note", x: ax, y: 5.8, w: aw, h: 0.4, fontSize: 12, color: C.accent2 });

    // Middle: what runs on the developer's machine
    const bx = 4.2, bw = 5.35, pad = 0.25, ix = bx + pad, iw = bw - 2 * pad;
    label(s, "DEVELOPER'S MACHINE", bx, 1.55, bw);
    card(s, "machine", { x: bx, y: 1.95, w: bw, h: 4.25 });
    s.addShape(pres.ShapeType.roundRect, { objectName: "session card", x: ix, y: 2.15, w: iw, h: 1.15, rectRadius: 0.08,
      fill: { color: C.background1 }, line: { color: C.accent3, width: 1 } });
    text(s, "One session, one expert", { objectName: "session title", x: ix + 0.22, y: 2.27, w: iw - 0.44, h: 0.34, fontSize: 18, bold: true, color: C.text1 });
    text(s, "A Claude Agent SDK session, loaded with that expert's prose and nothing else", {
      objectName: "session text", x: ix + 0.22, y: 2.63, w: iw - 0.44, h: 0.55, fontSize: 14, color: C.text2 });

    text(s, "Shared tools", { objectName: "tools title", x: ix, y: 3.47, w: iw, h: 0.3, fontSize: 14, bold: true, color: C.text2 });
    const tools = ["start", "measure", "query", "checked", "keep", "undo", "ask", "call"];
    const cg = 0.12, cw = (iw - 3 * cg) / 4, ch = 0.36;
    tools.forEach((tool, i) => {
      const x = ix + (i % 4) * (cw + cg), y = 3.83 + Math.floor(i / 4) * (ch + 0.1);
      s.addShape(pres.ShapeType.roundRect, { objectName: `tool ${tool}`, x, y, w: cw, h: ch, rectRadius: 0.18,
        fill: { color: C.background1 }, line: { color: C.accent3, width: 1 } });
      text(s, tool, { objectName: `tool ${tool} text`, x, y, w: cw, h: ch, fontSize: 12, align: "center", valign: "middle", color: C.text2 });
    });

    s.addShape(pres.ShapeType.roundRect, { objectName: "gate card", x: ix, y: 4.82, w: iw, h: 1.2, rectRadius: 0.08,
      fill: { color: C.text1 }, line: { type: "none" } });
    s.addShape(pres.ShapeType.ellipse, { objectName: "gate marker", x: ix + 0.24, y: 5.03, w: 0.17, h: 0.17,
      fill: { color: C.accent1 }, line: { type: "none" } });
    text(s, "The gate", { objectName: "gate title", x: ix + 0.52, y: 4.94, w: iw - 0.74, h: 0.34, fontSize: 18, bold: true, color: C.background1 });
    text(s, "Keeps a change only when it is cheaper past the noise and the answers still hold", {
      objectName: "gate text", x: ix + 0.22, y: 5.32, w: iw - 0.44, h: 0.6, fontSize: 14, color: C.accent3 });

    // Right: the team's agent, and what comes out
    const cx = 10.15, cwid = 2.6;
    label(s, "THE TEAM'S AGENT", cx, 1.55, cwid);
    card(s, "agent card", { x: cx, y: 1.95, w: cwid, h: 1.75 });
    text(s, "LangGraph agent", { objectName: "agent title", x: cx + 0.2, y: 2.09, w: cwid - 0.4, h: 0.34, fontSize: 18, bold: true, color: C.text1 });
    text(s, "Runs in its own environment. A probe records every model call", {
      objectName: "agent text", x: cx + 0.2, y: 2.47, w: cwid - 0.4, h: 1.1, fontSize: 14, color: C.text2 });
    card(s, "output card", { x: cx, y: 4.45, w: cwid, h: 1.75 });
    text(s, "What comes out", { objectName: "output title", x: cx + 0.2, y: 4.59, w: cwid - 0.4, h: 0.34, fontSize: 18, bold: true, color: C.text1 });
    text(s, "A report, a branch with the change, and the record of the run", {
      objectName: "output text", x: cx + 0.2, y: 4.97, w: cwid - 0.4, h: 1.1, fontSize: 14, color: C.text2 });

    // How they connect
    arrow(s, "catalogue to machine", ax + aw + 0.09, 3.81, bx - (ax + aw) - 0.18);
    arrow(s, "session and agent", bx + bw + 0.09, 2.72, cx - (bx + bw) - 0.18, true);
    arrow(s, "gate to output", bx + bw + 0.09, 5.32, cx - (bx + bw) - 0.18);

    text(s, "The expert decides how to work. Code decides what counts as proven.", {
      objectName: "takeaway", x: 0.6, y: 6.5, w: 12.1, h: 0.34, fontSize: 16, color: C.text1, valign: "middle" });
    s.addNotes("Three parts. On the left, the catalogue: each expert is a folder of markdown, fetched when a developer asks for it. "
      + "In the middle, the developer's machine: one Claude Agent SDK session runs as one expert and works through shared tools. "
      + "The gate is code: it keeps a change only when the measured cost is down past the noise and a separate reader finds the answers still hold. "
      + "On the right, the team's own agent, run in its own environment under a probe that records every model call. "
      + "An expert can name a tool or a rule; it cannot bring its own, so it cannot loosen its own proof.");
  }

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
