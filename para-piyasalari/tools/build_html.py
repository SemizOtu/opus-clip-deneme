"""Inject the timeline (and the mixed soundtrack, when present) into the page.

usage: python3 build_html.py <src_html> <timeline_json> <build_dir> [audio_mp3]
Writes <build_dir>/artifact.html (published page) and <build_dir>/capture.html (offline frame renderer).
"""
import base64, json, os, re, sys

src, tl_path, out = sys.argv[1:4]
audio = sys.argv[4] if len(sys.argv) > 4 else None
html = open(src, encoding="utf-8").read()
tl = json.dumps(json.load(open(tl_path, encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))
html = html.replace("/*@TIMELINE@*/null", tl, 1)
if audio and os.path.exists(audio):
    html = html.replace('/*@AUDIO@*/""', '"' + base64.b64encode(open(audio, "rb").read()).decode() + '"', 1)
open(os.path.join(out, "artifact.html"), "w", encoding="utf-8").write(html)

engine = re.search(r'<script id="pp-engine">(.*?)</script>', html, re.S).group(1)
fonts = open(os.path.join(out, "fonts", "local.css"), encoding="utf-8").read()
cap = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><title>capture</title>
<style>{fonts}
html,body{{margin:0;background:#000;overflow:hidden}} canvas{{display:block;width:1920px;height:1080px}}</style></head>
<body><canvas id="cv" width="1920" height="1080"></canvas><script>window.__PP_CAPTURE = true;</script>
<script>{engine}</script></body></html>"""
open(os.path.join(out, "capture.html"), "w", encoding="utf-8").write(cap)
print("artifact.html", os.path.getsize(os.path.join(out, "artifact.html")), "bytes")
