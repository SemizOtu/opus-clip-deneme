# Seni duyuyorum — motion animasyon

Romantik ilişkilerdeki tartışma döngüsünü (biri suçlar, öteki susar) ve döngüyü kırmanın dört adımını anlatan ~50 saniyelik animasyon.
Çıktı: `out/seni-duyuyorum.mp4` (1920×1080, 60 fps, AAC). Sorun, çözüm, seslendirme metni ve sahne/duygu planı: `PLAN.md`.

## Yeniden üretme

```bash
# 1. Seslendirme: her cümle parçası ayrı seslendirilir (birden çok kayıt, Whisper ile en anlaşılırı), parçalar 0,34 sn nefesle birleşir
python3 tools/tts_parts.py <vits-piper-tr_TR-fahrettin-medium> <sherpa-onnx-whisper-small> build/vo 10 .34
#    yalnızca bazı sahneleri yeniden seslendirmek için: ... build/vo 16 .34 s5,s7   (takes=0 sadece yeniden birleştirir)
# 2. Zaman çizelgesi (parça bazında kelime zamanları ve sahne işaretleri)
python3 tools/build_timeline.py build/vo build/timeline.json
# 3. Duyguya göre değişen müzik ve ortam sesleri + seslendirme miksajı, -15 LUFS
python3 tools/build_audio.py build/timeline.json build/vo build/mix.wav
tools/loudnorm.sh build/mix.wav build/mix_norm.wav
ffmpeg -i build/mix_norm.wav -c:a libmp3lame -b:a 160k -ar 44100 build/mix.mp3
# 4. Sayfalar ve video
python3 tools/build_html.py src/seni-duyuyorum.html build/timeline.json build build/mix.mp3
node tools/render.mjs video build/capture.html build/master.mp4 build/mix_norm.wav 3
# 5. Kağıt dokusu H.264'te çok yer kapladığı için master hafifçe yumuşatılıp yeniden sıkıştırılır (217 MB → 36 MB)
ffmpeg -i build/master.mp4 -vf "hqdn3d=1.2:1.2:5:5" -c:v libx264 -preset slow -crf 20 -profile:v high -x264-params aq-mode=3 \
  -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a copy -movflags +faststart out/seni-duyuyorum.mp4
```

Gerekenler `../para-piyasalari/README.md` ile aynıdır. `build/fonts/`: Newsreader, Figtree ve Caveat (SIL Open Font License).
