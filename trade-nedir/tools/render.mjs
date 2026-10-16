// Render frames of build/capture.html with headless Chromium.
//
//   node render.mjs stills <capture.html> <out_dir> <t1,t2,...>     one PNG per time (seconds)
//   node render.mjs frames <capture.html> <out_dir> [workers]       every frame at the timeline fps
//   node render.mjs video  <capture.html> <out.mp4> <audio.wav> [workers]   frames piped straight into ffmpeg
//
// Each worker owns a page and renders a contiguous slice of frames; the renderer is a pure
// function of time, so slices can be produced in any order.
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { spawn } from "node:child_process";

const require = createRequire("/opt/node22/lib/node_modules/");
const { chromium } = require("playwright");
const [mode, page_, outDir, arg, arg2] = process.argv.slice(2);
if (mode !== "video") mkdirSync(outDir, { recursive: true });
const url = pathToFileURL(resolve(page_)).href;

const browser = await chromium.launch({ args: ["--force-color-profile=srgb", "--disable-gpu-vsync"] });
async function openPage() {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on("pageerror", e => console.error("pageerror:", e.message));
  page.on("console", m => { if (m.type() === "error") console.error("console:", m.text()); });
  await page.goto(url);
  await page.waitForFunction(() => window.PP);
  await page.evaluate(() => window.PP.ready);
  return page;
}
async function grab(page, t) {
  const b64 = await page.evaluate(t => { window.PP.render(t); return document.getElementById("cv").toDataURL("image/png").slice(22); }, t);
  return Buffer.from(b64, "base64");
}

const t0 = Date.now();
if (mode === "stills") {
  const page = await openPage();
  for (const t of arg.split(",").map(Number)) writeFileSync(`${outDir}/t${t.toFixed(2)}.png`, await grab(page, t));
} else if (mode === "video") {
  const workers = Number(arg2 || 4);
  const probe = await openPage();
  const { duration, fps } = await probe.evaluate(() => ({ duration: window.PP.duration, fps: window.PP.fps }));
  await probe.close();
  const total = Math.ceil(duration * fps);
  const ff = spawn("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y",
    "-f", "image2pipe", "-framerate", String(fps), "-c:v", "png", "-i", "-", "-i", arg,
    "-map", "0:v", "-map", "1:a",
    "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high", "-x264-params", "aq-mode=3",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", outDir], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise(r => ff.on("close", r));
  const pending = new Map(); let next = 0;
  const flush = async () => {
    while (pending.has(next)) {
      const buf = pending.get(next); pending.delete(next); next++;
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
      if (next % 120 === 0) console.log(`${next}/${total} frames, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
  };
  let chain = Promise.resolve();
  await Promise.all(Array.from({ length: workers }, async (_, w) => {
    const page = await openPage();
    for (let f = w; f < total; f += workers) {
      while (f - next > workers * 6) await new Promise(r => setTimeout(r, 5));
      pending.set(f, await grab(page, f / fps));
      chain = chain.then(flush);
    }
  }));
  await chain; ff.stdin.end();
  const code = await done;
  console.log(`total ${total} frames, ffmpeg exit ${code}`);
} else {
  const workers = Number(arg || 4);
  const probe = await openPage();
  const { duration, fps } = await probe.evaluate(() => ({ duration: window.PP.duration, fps: window.PP.fps }));
  await probe.close();
  const total = Math.ceil(duration * fps);
  let done = 0;
  await Promise.all(Array.from({ length: workers }, async (_, w) => {
    const page = await openPage();
    for (let f = w; f < total; f += workers) {
      writeFileSync(`${outDir}/f${String(f).padStart(5, "0")}.png`, await grab(page, f / fps));
      if (++done % 120 === 0) console.log(`${done}/${total} frames, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
  }));
  console.log(`total ${total} frames`);
}
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
await browser.close();
