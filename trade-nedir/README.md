# Trade nedir? — motion animasyon

Trade'i yedi sahnede anlatan 33 saniyelik animasyon: tanım, emir defteri, mumun anatomisi, trend, işlem planı, risk yönetimi, sistem.
Çıktı: `out/trade-nedir.mp4` (1920×1080, 60 fps, AAC). Görsel dil: siyah, şampanya altını ve fildişi; sahne geçişleri pop art
(Ben-Day noktaları, çizgi roman patlamaları, Warhol ızgarası, çizgi roman panelleri, STOP! tabelası).

Görüntü `src/trade-nedir.html` içindeki canvas motorunda zamanın saf bir fonksiyonu olarak çizilir; seslendirme, müzik ve
efektler `build/timeline.json` içindeki aynı zaman çizelgesine bağlıdır. Tüm fiyatlar ve grafikler temsilidir; eğitim amaçlıdır, yatırım tavsiyesi değildir.

## Yeniden üretme

```bash
# 1. Seslendirme: her cümle için 9 kayıt, Whisper ile en anlaşılırı seçilir ("say" alanı telaffuz içindir: Trade → Treyd)
python3 tools/tts_takes.py <vits-piper-tr_TR-fahrettin-medium> <sherpa-onnx-whisper-small> build/vo 9
# 2. Zaman çizelgesi: sahne süreleri, kelime zamanları, geçiş pencereleri
python3 tools/build_timeline.py build/vo build/timeline.json
# 3. Müzik (A minör lounge, 100 BPM, FM elektrik piyano) + efektler + seslendirme miksajı, -15 LUFS
python3 tools/build_audio.py build/timeline.json build/vo build/mix.wav
tools/loudnorm.sh build/mix.wav build/mix_norm.wav
ffmpeg -i build/mix_norm.wav -c:a libmp3lame -b:a 160k -ar 44100 build/mix.mp3
# 4. Sayfalar ve video
python3 tools/build_html.py src/trade-nedir.html build/timeline.json build build/mix.mp3
node tools/render.mjs video build/capture.html out/trade-nedir.mp4 build/mix_norm.wav 3
```

Gerekenler `../para-piyasalari/README.md` ile aynıdır. `build/fonts/`: Cormorant Garamond, Manrope, JetBrains Mono ve Bangers (SIL Open Font License).
