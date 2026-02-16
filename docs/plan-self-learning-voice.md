# LiveClaw — Self-Learning Voice Architecture Plan
# Tarih: 2026-02-16

## Vizyon

Kullandıkça öğrenen, zamanla AI'a ihtiyaç duymadan çalışan bir canlı sesli konuşma sistemi.

---

## 1. Self-Learning Ses Arşivi

### Başlangıç (İlk günler)
- Her mesaj → TTS API → ses üret → oynat → arşive kaydet + AI ile indexle
- %0 arşiv, %100 TTS

### Öğrenme (1 hafta)
- Tekrarlayan mesajlar arşivden gelmeye başlar
- %40 arşiv, %60 TTS

### Olgunluk (1-3 ay)
- Çoğu mesaj arşivden eşleşir
- %80-95 arşiv, %5-20 TTS

### Eşleştirme Zinciri

```
Gelen mesaj
    │
    ├─ 1. Regex/Fuzzy (< 1ms)
    │   Arşivdeki keyword/pattern'larla eşleştir
    │   Bulursa → arşivden ses çal ✓
    │
    ├─ 2. Gemini Flash + Arşiv Context (< 500ms)
    │   Arşiv manifest'ini context olarak ver
    │   "Bu mesaja en uygun arşiv kaydı hangisi?"
    │   Bulursa → arşivden ses çal ✓
    │   + regex/fuzzy eşleşme sistemini bu mesaj için eğit
    │   (yeni keyword ekle, pattern güncelle)
    │
    ├─ 3. Arşivlenebilir mi? (Gemini Flash karar verir)
    │   Mesaj jenerik mi yoksa çok niş terim mi içeriyor?
    │   
    │   ├─ Arşivlenebilir → TTS ile yeni ses üret
    │   │   → Arşive kaydet
    │   │   → AI ile keyword'le, indexle
    │   │   → Regex/fuzzy pattern'ları güncelle
    │   │
    │   └─ Arşivlenemez (çok niş) → TTS ile ses üret, oynat, arşivleme
    │       → Opsiyonel: OpenClaw eğitim dosyasını güncelle
    │         "Ara mesajlarda niş terimler kullanma"
    │
    └─ 4. TTS Fallback
        Her durumda ses üretilir ve oynatılır
```

### Auto-Indexing (Yeni Ses Arşive Eklendiğinde)

```
Yeni ses oluştu
    │
    ├─ AI (veya regex) ile keyword çıkar
    │   Örnek: "3 kaynak buldum, analiz ediyorum"
    │   → keywords: ["kaynak", "buldum", "analiz"]
    │   → variations: ["sonuç buldum", "kaynakları inceliyorum", ...]
    │   → regex pattern: r"(\d+)\s*(kaynak|sonuç)\s*(buldum|bulundu)"
    │
    ├─ Fuzzy match threshold belirle
    │   → base_text: "3 kaynak buldum, analiz ediyorum"
    │   → threshold: 75%
    │
    └─ Arşiv manifest'ini güncelle
        → library_manifest.json'a yeni entry ekle
```

---

## 2. Mesaj Yakalama (Proxy Yaklaşımı)

### Mevcut Sistem (Sorunlu)
```
OpenClaw → Telegram API → mesaj gönderilir → bildirim gelir
→ LiveClaw mesajı görür → siler → ses gönderir
Sorun: çift bildirim, silme gecikmesi
```

### Önerilen Sistem (Proxy)
```
OpenClaw → [Telegram API Proxy] → mesajı YAKALA
→ OpenClaw'a "gönderildi" fake response dön
→ Mesaj içeriğini LiveClaw'a ilet
→ LiveClaw sese çevirip gerçek Telegram API'ye gönderir
Sonuç: sıfır çift bildirim, sıfır silme
```

### Nasıl Yapılır
- Küçük bir HTTP proxy yazılır (FastAPI/aiohttp)
- OpenClaw'un bot token'ının API base URL'ini bu proxy'ye yönlendir
- Proxy sendMessage/sendVoice çağrılarını yakalar
- Text mesajları LiveClaw'a iletir, voice mesajları direkt geçirir
- OpenClaw'a başarılı response döner

### Avantajlar
- OpenClaw'da sıfır değişiklik gerekir
- Çift bildirim tamamen ortadan kalkar
- Mesaj silme gereksiz olur
- Daha temiz ve güvenilir mimari

---

## 3. Dual-Track Orkestrasyon (OpenClaw Tarafı)

### Track A: Fast Path (< 500ms)
- Gemini 2.5 Flash Lite veya benzer hızlı model
- Kullanıcı mesaj gönderdiğinde anında contextual ack üretir
- "Hmm güzel soru, bakıyorum..."
- LiveClaw bunu ses arşivinden anında çalar

### Track B: Slow Path (5-30s)
- Opus 4.6 (thinking model)
- Derin araştırma, tool calling, subagent delegation
- Streaming progress mesajları gönderir
- Final cevap gönderir

### Senkronizasyon
- Track A filler gönderir → LiveClaw arşivden çalar
- Track B progress gönderir → LiveClaw arşivden veya TTS ile çalar
- Dedup: aynı intent iki kez seslendirmez

---

## 4. OpenClaw Skill: Arşivlenebilir Cevaplar

OpenClaw'a skill eklenerek ara mesajlarda:
- Kısa, jenerik, tekrar kullanılabilir ifadeler kullanması sağlanır
- ❌ "Quantum entanglement ile ilgili 3 makale buldum"
- ✅ "3 kaynak buldum, analiz ediyorum"
- İkincisi arşivlenebilir, birincisi çok niş

---

## 5. Default Filler Sesler (Seed Arşiv)

Sistem başlarken 20-30 filler ses önceden üretilir:
- "Tamam bakıyorum"
- "Bi düşüneyim"
- "Hmm güzel soru"
- "Araştırıyorum"
- "Buldum, şimdi özetliyorum"
- "Tamamlandı"
- vb.

Bu arşivin ilk seed'i olur. Sistem kullandıkça büyür.

---

## 6. Fizibilite

| Özellik | Mümkün | Zorluk | Öncelik |
|---------|--------|--------|---------|
| Self-learning ses arşivi | ✅ | Orta | Yüksek |
| Mesaj yakalama proxy | ✅ | Orta | Yüksek |
| Arşivlenebilir cevap skill | ✅ | Düşük | Orta |
| Gemini Flash fallback eşleştirme | ✅ | Düşük | Yüksek |
| Auto-indexing (AI keyword) | ✅ | Düşük | Yüksek |
| Regex/fuzzy self-training | ✅ | Orta | Yüksek |
| Default filler sesler | ✅ | Çok düşük | Çok yüksek |
| Dual-track orkestrasyon | ✅ | Orta | Yüksek |
