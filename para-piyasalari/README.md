# Para Piyasaları — motion animasyon

Para piyasalarını altı sahnede anlatan 22,5 saniyelik animasyon. Çıktı: `out/para-piyasalari.mp4` (1920×1080, 60 fps, AAC).

Görüntü `src/para-piyasalari.html` içindeki canvas motorunda zamanın saf bir fonksiyonu olarak çizilir; seslendirme, müzik ve efektler `build/timeline.json` içindeki aynı zaman çizelgesine bağlıdır.

## Yeniden üretme

```bash
# 1. Seslendirme (Piper "Fahrettin" sesi, sherpa-onnx). Her cümle için 9 kayıt, Whisper ile en anlaşılırı seçilir.
python3 tools/tts_takes.py <vits-piper-tr_TR-fahrettin-medium> <sherpa-onnx-whisper-small> build/vo 9
# 2. Zaman çizelgesi: sahne süreleri, kelime zamanları, geçiş işaretleri
python3 tools/build_timeline.py build/vo build/timeline.json
# 3. Müzik + efekt sentezi, seslendirme miksajı, -15 LUFS normalizasyon
python3 tools/build_audio.py build/timeline.json build/vo build/mix.wav
tools/loudnorm.sh build/mix.wav build/mix_norm.wav
ffmpeg -i build/mix_norm.wav -c:a libmp3lame -b:a 160k -ar 44100 build/mix.mp3
# 4. Sayfalar: build/artifact.html (oynatıcı, ses gömülü) ve build/capture.html (kare işleyici)
python3 tools/build_html.py src/para-piyasalari.html build/timeline.json build build/mix.mp3
# 5. Video: kareler Chromium'da çizilip doğrudan ffmpeg'e akıtılır
node tools/render.mjs video build/capture.html out/para-piyasalari.mp4 build/mix_norm.wav 3
```

Gerekenler: Python 3 (numpy, scipy, soundfile, sherpa-onnx), Node 22 + Playwright Chromium, ffmpeg (libx264, libmp3lame).
Model dosyaları sherpa-onnx GitHub sürümlerinden indirilir (`tts-models/vits-piper-tr_TR-fahrettin-medium`, `asr-models/sherpa-onnx-whisper-small`).
`build/fonts/` Google Fonts'tan alınmış Bodoni Moda ve IBM Plex (SIL Open Font License) dosyalarıdır.

Faiz koridoru grafiğindeki değerler temsilidir.
