# OpenClaw — Anlık Cevap + Derin İşlem Orkestrasyon Araştırması v2
# Tarih: 2026-02-16

## Problem

OpenClaw Opus 4.6 kullanıyor. TTFT 10-15 saniye.
Streaming mesajlar geliyor: "araştırma başladı" → "sonuçlar geldi" → "özet çıkarıyorum" → final cevap
Bu streaming mesajlar bile 10-15s geç geliyor.
LiveClaw bu mesajları sese çevirmek istiyor + arşivle anlık hale getirmek istiyor.

---

## Endüstri Çözümleri

### 1. Speculative Tool Calling (GetStream — En İyi Kaynak)
Kaynak: https://getstream.io/blog/speculative-tool-calling-voice/

İki paralel track:
- Track A (Filler): Anında konuşma acknowledgment → TTS'e gönder
- Track B (Speculation): Sessizce tool prediction + execution

```python
async def handle_voice_input(user_audio):
    text = await transcribe(user_audio)
    
    # Paralel başlat
    filler_task = asyncio.create_task(generate_filler(text))
    tool_task = asyncio.create_task(predict_and_execute_tool(text))
    
    # Filler'ı hemen ses olarak gönder
    async for chunk in stream_from_task(filler_task):
        yield to_audio(chunk)
    
    # Tool sonucu (muhtemelen zaten bitti)
    tool_result = await tool_task
    
    # Final cevap
    final_answer = await generate_answer(text, tool_result)
    yield to_audio(final_answer)
```

### 2. Two-Stage Architecture (Fast Router + Slow LLM)
Kaynak: https://medium.com/@yuxiaojian/building-responsive-ai-7364e12937af

| Stage  | Model              | Latency     | Amaç              |
|--------|--------------------|-------------|--------------------|
| Router | 8B model / classifier | ~50-100ms | Intent detect, tool predict |
| Main   | Opus 4.6           | ~500ms+     | Derin cevap üret   |

Her ikisi aynı anda başlar:
- Router: intent belirler, tool çağırır
- Main LLM: konuşma cevabı üretir
- Router sonucu Main LLM'in context'ine enjekte edilir

### 3. Hybrid SLM + LLM Approach
Kaynak: arxiv.org/abs/2506.02153 "Small Language Models are the Future of Agentic AI"

- SLM (Gemini Flash, Haiku): Hızlı, tekrarlayan görevler (routing, summarization, extraction)
- LLM (Opus): Derin reasoning, karmaşık problem çözme
- SLM router olarak çalışır, basit istekleri kendisi cevaplar
- Karmaşık istekleri LLM'e yönlendirir

### 4. Parallel Guardrails + Fast Path
Kaynak: OpenAI "A Practical Guide to Building Agents" (cdn.openai.com)

- Guardrail (hafif SLM): Input validation, intent detection — paralel çalışır
- Sorun varsa hemen "fail fast" — pahalı model çağrılmaz
- Sorun yoksa ana model devam eder

### 5. ConvoCache — Semantic Response Caching
Kaynak: arxiv.org/abs/2406.18133 (Interspeech 2024)

- Gelen prompt'a semantik olarak benzer geçmiş prompt bul
- Cache'deki cevabı yeniden kullan
- %89 cache hit rate, 214ms ortalama latency
- LLM + TTS bypass

### 6. Prompt Caching (Anthropic/OpenAI)
- Büyük system prompt'lar için cache_control: ephemeral
- TTFT %70'e kadar azalır
- Tekrarlayan prefix'ler için ideal

---

## OpenClaw İçin Önerilen Mimari

### Dual-Track Orchestration

```
Kullanıcı mesaj gönderir
         │
         ├─── Track A: FAST PATH (< 500ms) ──────────────────────┐
         │    Gemini 2.5 Flash Lite                               │
         │    - Intent classification                             │
         │    - Contextual filler üret                            │
         │    - "Hmm, quantum fizik hakkında güzel soru!          │
         │       Araştırıyorum..."                                │
         │    → Telegram'a gönder                                 │
         │    → LiveClaw sese çevirir                             │
         │                                                        │
         ├─── Track B: SLOW PATH (5-30s) ─────────────────────────┤
         │    Opus 4.6 (thinking)                                 │
         │    - Derin araştırma                                   │
         │    - Tool calling                                      │
         │    - Subagent delegation                               │
         │                                                        │
         │    Streaming progress mesajları:                       │
         │    T+5s:  "Araştırma başladı"                          │
         │    T+10s: "3 kaynak buldum"                            │
         │    T+20s: "Özet çıkarıyorum"                          │
         │    T+30s: Final cevap                                  │
         │    → Her biri Telegram'a gönderilir                    │
         │    → LiveClaw her birini sese çevirir                  │
         │                                                        │
         └─── Track C: CACHE PATH (< 1ms) ────────────────────────┘
              Semantic cache check
              - Benzer soru daha önce soruldu mu?
              - Evet → cache'den cevap (Track A+B bypass)
              - Hayır → Track A + B devam
```

### Streaming Mesaj Senkronizasyonu

Problem: Track A filler gönderir, Track B de progress gönderir → çakışma?

Çözüm: Message Queue + Dedup

```python
class MessageOrchestrator:
    def __init__(self):
        self.sent_intents = set()  # "ack", "progress_search", "progress_found", "final"
    
    async def send_if_new(self, intent: str, text: str):
        """Aynı intent'i iki kez gönderme"""
        if intent in self.sent_intents:
            return
        self.sent_intents.add(intent)
        await bot.send_message(chat_id, text)
    
    async def handle_message(self, user_text: str):
        # Track C: Cache check
        cached = await semantic_cache.find(user_text)
        if cached:
            await self.send_if_new("final", cached)
            return
        
        # Track A: Fast ack (Gemini Flash)
        fast_task = asyncio.create_task(self.fast_ack(user_text))
        
        # Track B: Deep processing (Opus)
        slow_task = asyncio.create_task(self.deep_process(user_text))
        
        # Track A biter → hemen gönder
        ack = await fast_task
        await self.send_if_new("ack", ack)
        
        # Track B streaming progress
        async for progress in slow_task:
            if progress.type == "progress":
                await self.send_if_new(progress.intent, progress.text)
            elif progress.type == "final":
                await self.send_if_new("final", progress.text)
                # Cache'e kaydet
                await semantic_cache.store(user_text, progress.text)
```

### LiveClaw Entegrasyonu

LiveClaw tarafında:
1. Gelen mesajın tipini belirle (filler/progress/final)
2. Filler → ses arşivinden çal (< 50ms)
3. Progress → ses arşivinden çal veya kısa TTS (< 500ms)
4. Final → TTS ile sese çevir

```python
# LiveClaw interceptor'da
FILLER_PATTERNS = {
    r"araştırıyorum|researching": "ack_searching",
    r"başlıyorum|starting": "ack_started",
    r"buldum|found": "ack_found",
    r"özet|summary": "ack_summarizing",
    r"tamamlandı|done|bitti": "ack_done",
}

async def on_bot_message(text):
    # Ses arşivinde eşleşme var mı?
    clip = match_archive(text)
    if clip:
        play(clip)  # < 50ms
    else:
        audio = await tts.generate(text)  # 200-500ms
        play(audio)
```

---

## Model Latency Karşılaştırması

| Model                  | TTFT        | Maliyet (1M input) | Rol                |
|------------------------|-------------|---------------------|--------------------|
| Gemini 2.5 Flash Lite  | ~150-300ms  | ~$0.075             | Fast ack, routing  |
| Gemini 2.5 Flash       | ~200-400ms  | ~$0.15              | Smart ack          |
| Claude Haiku 4.5       | ~400-700ms  | ~$0.80              | Context-aware ack  |
| GPT-4.1-nano           | ~200-400ms  | ~$0.10              | Cheap ack          |
| Claude Opus 4.6        | ~3-15s      | ~$15.00             | Deep thinking      |

---

## Mevcut Framework'ler

### 1. LangChain RunnableParallel
- İki model'i aynı anda çalıştır
- Sonuçları birleştir
- URL: https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableParallel.html

### 2. AutoGen (Microsoft)
- Multi-agent async execution
- Farklı model'lerle farklı agent'lar
- URL: https://github.com/microsoft/autogen

### 3. LangGraph
- Graph-based orchestration
- Parallel node execution
- Conditional branching

### 4. Pipecat
- Voice pipeline framework
- STT → LLM → TTS zinciri
- Filler audio desteği
- URL: https://github.com/pipecat-ai/pipecat

### 5. Azure AI Agent Orchestration Patterns
- Sequential, Concurrent, Group Chat, Handoff, Magentic
- URL: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns

---

## Telegram API Teknikleri

### 1. Send-Then-Edit Pattern
```python
# Hemen placeholder gönder
msg = await bot.send_message(chat_id, "⏳ Araştırıyorum...")
# Opus çalışsın
result = await opus.generate(prompt)
# Placeholder'ı güncelle
await bot.edit_message_text(chat_id, msg.message_id, result)
```

### 2. Typing Indicator Loop
```python
async def keep_typing(chat_id):
    while True:
        await bot.send_chat_action(chat_id, "typing")
        await asyncio.sleep(4)  # typing 5s sürer
```

### 3. Progressive Streaming Edit
```python
msg = await bot.send_message(chat_id, "▌")
buffer = ""
async for chunk in opus.stream(prompt):
    buffer += chunk
    if len(buffer) % 50 == 0:  # Her 50 karakterde güncelle
        await bot.edit_message_text(chat_id, msg.message_id, buffer + "▌")
# Final
await bot.edit_message_text(chat_id, msg.message_id, buffer)
```

### Kısıtlamalar
- editMessageText: sadece bot'un kendi mesajları
- Rate limit: ~30 req/s global
- MessageNotModified: aynı text ile edit → hata
- Mesaj boyutu: max 4096 karakter

---

## Kaynaklar

### Akademik
- ConvoCache (Interspeech 2024): https://arxiv.org/abs/2406.18133
- Small Language Models for Agentic AI: https://arxiv.org/abs/2506.02153
- Multi-Model Orchestration: https://arxiv.org/pdf/2512.22402

### Blog / Teknik
- Speculative Tool Calling: https://getstream.io/blog/speculative-tool-calling-voice/
- Building Responsive AI (Latency Guide): https://medium.com/@yuxiaojian/building-responsive-ai-7364e12937af
- Canonical AI Semantic Cache: https://canonical.chat/blog/voice_ai_caching
- AI Agent Routing (Botpress): https://botpress.com/blog/ai-agent-routing
- Context-Aware Routing: https://blog.poespas.me/posts/2025/03/08/improving-chatbot-response-time-with-context-aware-routing/
- Azure Agent Patterns: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
- LLM Orchestration Top 22: https://research.aimultiple.com/llm-orchestration/
- Multi-Agent Orchestration: https://www.onabout.ai/p/mastering-multi-agent-orchestration
- Parallel Agent Processing: https://www.kore.ai/blog/parallel-ai-agent-processing
- n8n Agent Frameworks: https://blog.n8n.io/ai-agent-orchestration-frameworks/

### Framework
- Pipecat: https://github.com/pipecat-ai/pipecat
- pipecat-tts-cache: https://github.com/omChauhanDev/pipecat-tts-cache
- AutoGen: https://github.com/microsoft/autogen
- LangGraph: https://github.com/langchain-ai/langgraph

---

## Sonuç ve Aksiyon Planı

### OpenClaw Tarafı (Skill/Özellik)
1. **Dual-Track Skill**: Gemini Flash ile anlık ack + Opus ile derin işlem
2. **Streaming Progress**: Opus çalışırken ara durum mesajları gönder
3. **Semantic Cache**: Tekrarlayan sorulara anında cevap
4. **Intent Router**: Basit sorular → Flash, karmaşık → Opus

### LiveClaw Tarafı (Middleware)
1. **Ses Arşivi Eşleştirme**: Filler/progress mesajlarını arşivden çal
2. **TTS Cache**: Aynı text → cache'den ses
3. **Dedup**: Aynı intent'i iki kez seslendirme
4. **Progressive Audio**: Streaming mesajları sırayla seslendir

### Senkronizasyon
- OpenClaw mesaj tipini metadata olarak gönderir (filler/progress/final)
- LiveClaw metadata'ya göre davranış belirler
- Veya: LiveClaw regex ile mesaj tipini otomatik algılar
