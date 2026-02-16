# LiveClaw + OpenClaw — Anlık Cevap Orkestrasyon Araştırması
# Tarih: 2026-02-16

## Problem

OpenClaw, Claude Opus 4.6 (thinking model) kullanıyor. TTFT 10-15 saniye.
Kullanıcı "merhaba" yazıyor → 15 saniye sessizlik → cevap geliyor.
Bu UX olarak kabul edilemez.

## Çözüm Seçenekleri

---

### Seçenek 1: Send-Then-Edit (En Basit, Önerilen)

OpenClaw tarafında:
1. Kullanıcı mesaj gönderir
2. Bot ANINDA "⏳ Tamam, araştırıyorum..." placeholder gönderir (~50ms)
3. sendChatAction("typing") loop başlar (her 4 saniyede bir)
4. Opus arka planda çalışır (10-15s)
5. Opus bitince editMessageText ile placeholder gerçek cevapla değiştirilir

Avantaj: Sıfır ek maliyet, 30 satır kod
Dezavantaj: Placeholder statik, context-aware değil

```python
@dp.message()
async def handle(message):
    placeholder = await message.reply("⏳ Tamam, araştırıyorum...")
    asyncio.create_task(process_and_edit(message.chat.id, placeholder.message_id, message.text))

async def process_and_edit(chat_id, msg_id, text):
    typing_task = asyncio.create_task(keep_typing(chat_id))
    response = await claude.messages.create(model="claude-opus-4-6", ...)
    typing_task.cancel()
    await bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=response.content[0].text)
```

### Seçenek 2: Dual-Model (Fast ACK + Slow Deep)

1. Kullanıcı mesaj gönderir
2. Gemini Flash (~300ms) ve Opus (~15s) AYNI ANDA başlatılır
3. Flash hemen döner → "Hmm, quantum fizik hakkında güzel soru! Araştırıyorum..."
4. Opus döner → editMessageText ile gerçek cevap

Avantaj: Contextual acknowledgment
Dezavantaj: +$0.001/request maliyet

```python
fast_task = asyncio.create_task(call_gemini_flash(prompt))
slow_task = asyncio.create_task(call_opus(prompt))
fast_ack = await fast_task  # ~300ms
placeholder = await message.reply(fast_ack)
full_response = await slow_task  # ~15s
await bot.edit_message_text(..., text=full_response)
```

### Seçenek 3: Streaming + Progressive Edit

1. Placeholder gönder
2. Opus streaming API kullan
3. Her 1 saniyede editMessageText ile güncelle (cursor efekti: "▌")
4. Stream bitince final edit

Not: Opus'un thinking phase'i hala 5-15s. İlk token gelene kadar placeholder gerekli.

### Seçenek 4: Hybrid (Production-Grade)

Seçenek 1 + 2 + 3 birleşik:
- T+0.0s → Statik placeholder
- T+0.3s → Flash'tan contextual ack (edit)
- T+5-15s → Opus token'ları gelmeye başlar (streaming edit)
- T+15s → Final cevap

---

## LiveClaw Entegrasyonu — Çakışma Problemi

### Problem
Eğer OpenClaw "Başlıyorum..." placeholder gönderirse VE LiveClaw bunu sese çevirirse,
kullanıcı aynı mesajı iki kez duyar.

### Çözüm Seçenekleri

#### A) Typing-Only (Önerilen)
- OpenClaw placeholder TEXT göndermez, sadece sendChatAction("typing") kullanır
- LiveClaw'ın intercept edeceği bir şey yok → çakışma yok
- LiveClaw kendi ses arşivinden "başlıyorum" filler'ı çalar
- OpenClaw sadece final cevabı gönderir → LiveClaw onu sese çevirir

#### B) Metadata Flag
- OpenClaw placeholder'a özel prefix ekler: "[FILLER] Araştırıyorum..."
- LiveClaw "[FILLER]" prefix'li mesajları atlar
- OpenClaw editMessageText ile final cevabı gönderir → LiveClaw onu işler

#### C) LiveClaw Owns Filler (En Temiz)
- LiveClaw kullanıcının mesajını görür → hemen ses arşivinden filler çalar
- OpenClaw hiçbir placeholder göndermez
- OpenClaw sadece final cevabı gönderir → LiveClaw sese çevirir
- Tek acknowledgment, tek kaynak, sıfır çakışma

---

## Model Latency Karşılaştırması (TTFT)

| Model                    | TTFT        | Maliyet (1M input) | Kullanım          |
|--------------------------|-------------|---------------------|--------------------|
| Gemini 2.5 Flash Lite   | ~150-300ms  | ~$0.075             | En hızlı ack       |
| Gemini 2.5 Flash        | ~200-400ms  | ~$0.15              | Hızlı ack          |
| Claude Haiku 4.5        | ~400-700ms  | ~$0.80              | Akıllı ack         |
| GPT-4.1-nano            | ~200-400ms  | ~$0.10              | Ucuz ack            |
| Claude Opus 4.6         | ~3-15s      | ~$15.00             | Derin düşünme       |

---

## Telegram API Kısıtlamaları

| Kısıt                        | Detay                                          |
|------------------------------|-------------------------------------------------|
| sendChatAction("typing")     | 5 saniye sürer, loop gerekli                    |
| editMessageText              | Sadece bot'un kendi mesajlarını düzenleyebilir   |
| Rate limit                   | ~30 request/saniye global                        |
| Mesaj boyutu                 | Max 4096 karakter                                |
| MessageNotModified hatası    | Aynı text ile edit yapılırsa fırlatılır          |

---

## Anthropic Optimizasyonları

1. Streaming: messages.stream() — TTFT azaltmaz ama progressive UX sağlar
2. Prompt Caching: Büyük system prompt'lar için cache_control: ephemeral
3. max_tokens: Gereksiz yüksek tutma
4. Model routing: Basit sorular → Haiku, karmaşık → Opus

---

## Kaynaklar

### Akademik / Blog
- ConvoCache (Interspeech 2024): https://arxiv.org/abs/2406.18133
- Canonical AI Semantic Cache: https://canonical.chat/blog/voice_ai_caching
- GetStream Speculative Tool Calling: https://getstream.io/blog/speculative-tool-calling-voice/
- Sierra Voice Latency: https://sierra.ai/blog/voice-latency
- AssemblyAI Low Latency Voice: https://www.assemblyai.com/blog/low-latency-voice-ai

### Framework / Araç
- Pipecat: https://github.com/pipecat-ai/pipecat
- pipecat-tts-cache: https://github.com/omChauhanDev/pipecat-tts-cache
- LangChain RunnableParallel: https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableParallel.html

### Telegram Patterns
- editMessageText: https://core.telegram.org/bots/api#editmessagetext
- sendChatAction: https://core.telegram.org/bots/api#sendchataction

---

## Sonuç ve Öneri

### Kısa Vadeli (Hemen Uygulanabilir)
OpenClaw'a "Send-Then-Edit" pattern ekle:
- Placeholder gönder → typing loop → Opus → edit
- LiveClaw'da "[FILLER]" prefix filtresi ekle

### Orta Vadeli
LiveClaw'a filler ses arşivi ekle:
- Kullanıcı mesaj gönderdiğinde LiveClaw hemen "başlıyorum" sesi çalar
- OpenClaw sadece final cevap gönderir
- Sıfır çakışma, en temiz UX

### Uzun Vadeli
Dual-model orkestrasyon:
- Gemini Flash contextual ack + Opus deep processing
- Streaming progressive edit
- LiveClaw semantic cache ile tekrarlayan cevapları anında ses olarak çalar
