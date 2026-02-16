# LiveClaw — Ses Arşivi Eşleştirme Sistemi Araştırma Notları
# Tarih: 2026-02-16

## 1. Mevcut Açık Kaynak Projeler

### Tier 1: Doğrudan İlgili

#### ConvoCache (Interspeech 2024 — Akademik Paper)
- arXiv: https://arxiv.org/abs/2406.18133
- Ne yapıyor: Spoken chatbot'lar için semantic caching. Gelen prompt'a semantik olarak
  benzer bir geçmiş prompt bulur → cache'deki cevabı yeniden kullanır.
- Sonuçlar: UniEval coherence %90 threshold ile promptların %89'unu cache'den cevaplayabiliyor.
  Ortalama latency: 214ms (LLM+TTS: 1s+)
- Yöntem: Embedding similarity (cosine) ile eşleştirme
- Durum: Akademik paper, kod yok ama yaklaşım açık

#### pipecat-tts-cache (GitHub, aktif)
- URL: https://github.com/omChauhanDev/pipecat-tts-cache
- PyPI: pip install pipecat-tts-cache
- Ne yapıyor: Pipecat framework için TTS caching mixin
- Yöntem: Exact string match (normalized text + voice_id + model)
- Cache hit: ~0.1ms (memory), ~2ms (Redis) vs API: 200-1500ms
- Backend: LRU memory veya Redis
- Durum: Aktif, 10 star

#### pipecat-speech-cache (GitHub)
- URL: https://github.com/steinathan/pipecat-speech-cache
- Ne yapıyor: Redis-based TTS response caching for Pipecat
- Durum: 3 star, alternatif implementasyon

#### Canonical AI — Context-Aware Semantic Cache
- URL: https://canonical.chat/blog/voice_ai_caching
- GitHub: https://github.com/Canonical-AI-Inc/canonical
- Ne yapıyor: Voice AI için context-aware semantic caching servisi
- Özellikler:
  - Embedding similarity ile cache hit/miss
  - Context-aware: konuşma geçmişini dikkate alıyor
  - Multi-tenant: farklı persona'lar için farklı cache bucket'ları
- Durum: Ticari ürün, yaklaşım açık

#### bebora/tts-cache (GitHub)
- URL: https://github.com/bebora/tts-cache
- Ne yapıyor: Google Cloud TTS için caching proxy
- Yöntem: Exact text match → MinIO'da cache
- Durum: 2 star, minimal

### Tier 2: Framework / Altyapı

#### Pipecat (GitHub, ~10k star)
- URL: https://github.com/pipecat-ai/pipecat
- Ne yapıyor: Open source voice pipeline framework (STT → LLM → TTS)
- Issue #2629: TTS caching tartışması aktif
- Kullanım: Pipeline'a "clip matcher" processor eklenebilir

#### LAION-AI/CLAP (~2k star)
- URL: https://github.com/LAION-AI/CLAP
- Ne yapıyor: Text-Audio joint embedding. Text query ile audio dosyası arasında
  similarity hesaplayabiliyor.
- PyPI: pip install laion-clap
- Kullanım: Ses kliplerini embed et, gelen text'i embed et, cosine similarity ile eşleştir

#### microsoft/CLAP (~640 star)
- URL: https://github.com/microsoft/CLAP
- Ne yapıyor: Microsoft'un CLAP implementasyonu
- PyPI: pip install msclap

### Tier 3: Yakın Projeler

#### Mycroft TTS Cache (Archived)
- URL: https://mycroft.ai/blog/mimic2-speed-boost-response-caching/
- Ne yapıyor: Stock phrase'ler için TTS çıktısını cache'liyor
- Yöntem: Exact string match
- Sonuç: %25 latency azalması
- Durum: Mycroft kapandı, OVOS devam ediyor

#### Picovoice/Rhino (~698 star)
- URL: https://github.com/Picovoice/rhino
- Ne yapıyor: On-device Speech-to-Intent engine
- Kullanım: Intent classification katmanı olarak kullanılabilir

#### SoundBot
- URL: https://github.com/Mike014/SoundBot
- Ne yapıyor: Intent-based dictionary lookup → ses çalma (müzik için)
- Yöntem: NLTK ile intent extraction → dictionary key match → play audio

### Tier 4: VTuber / Streamer Projeleri (TTS-based, clip matching yok)

- Open-LLM-VTuber (~5k star): https://github.com/Open-LLM-VTuber/Open-LLM-VTuber
- AI-Waifu-Vtuber: https://github.com/ardha27/AI-Waifu-Vtuber
- voice-chat-ai: https://github.com/bigsk1/voice-chat-ai

### Tier 5: Discord Soundboard Botları (Exact command match)

- discord-soundbot (~197 star): https://github.com/markokajzer/discord-soundbot
- Telegram audio bot: https://github.com/doublebon/Telegram_audio_bot

---

## 2. Eşleştirme Yaklaşımları Karşılaştırması

| Yaklaşım              | Latency (500 klip) | Doğruluk        | Türkçe | Bellek  | Paraphrase |
|------------------------|---------------------|------------------|--------|---------|------------|
| Regex/Exact            | < 0.1ms             | Düşük (birebir)  | ✓      | ~0      | ✗          |
| Fuzzy (rapidfuzz)      | < 1ms               | Orta-Yüksek      | ✓      | < 1MB   | ✗          |
| TF-IDF/BM25            | < 1ms               | Orta             | Stemmer| < 5MB   | ✗          |
| SBERT Embedding        | 5-15ms              | Yüksek           | ✓      | ~500MB  | ✓          |
| CLAP (text-audio)      | 10-20ms             | Yüksek           | ?      | ~600MB  | ✓          |
| scikit-learn (Rasa)    | < 1ms               | Yüksek           | ✓      | < 5MB   | Kısmen     |
| N-gram/MinHash         | 1-3ms               | Orta             | ✓      | < 1MB   | ✗          |

---

## 3. Önerilen Mimari: 4 Katmanlı Hibrit (ConvoCache + Rasa inspired)

```
Gelen mesaj
    │
    ├─ Katman 1: Exact + Regex (< 0.1ms)
    │   Dict lookup + compiled regex patterns
    │   Confidence: 1.0
    │
    ├─ Katman 2: Fuzzy Match — rapidfuzz (< 1ms)
    │   token_set_ratio ile tüm varyasyonlara karşı
    │   Threshold: %85+
    │
    ├─ Katman 3: Semantic — SBERT embedding (5-15ms)
    │   paraphrase-multilingual-MiniLM-L12-v2
    │   Cosine similarity, threshold: %65+
    │   Sadece Katman 1-2 tutmadığında çalışır
    │
    └─ Katman 4: TTS Fallback
        Hiçbiri tutmadı → ses üret ve cache'e kaydet
```

---

## 4. Klip Metadata Yapısı

```json
{
    "id": "ack_started",
    "audio_file": "ack_started.ogg",
    "description": "Göreve başlama onayı",
    "keywords": ["tamam", "başlıyorum", "bakıyorum"],
    "variations": [
        "tamam başlıyorum",
        "peki hemen bakıyorum",
        "ok let me check",
        "olur bakayım"
    ],
    "priority": 1
}
```

---

## 5. Anahtar Çıkarımlar

1. Bizim yaptığımız şeyi yapan hazır proje YOK — bu novel bir kombinasyon
2. ConvoCache paper'ı en yakın akademik referans (semantic cache for spoken chatbots)
3. Building block'lar hazır: rapidfuzz, sentence-transformers, CLAP, scikit-learn
4. pipecat-tts-cache mimari olarak cache katmanının temeli olabilir
5. Canonical AI'ın context-aware yaklaşımı gelecekte eklenebilir
6. Mesajların %70-80'i Katman 1-2'de çözülür (< 1ms)
7. Katman 4'te üretilen sesler otomatik cache'e eklenerek sistem zamanla öğrenir
