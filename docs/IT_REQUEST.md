# Požadavek na LLM infrastrukturu pro workshop SPRINT

## Kontext

**Aplikace SPRINT** je webová aplikace pro hodnocení českých právních textů podle jazykových pravidel. Uživatelé vloží/upraví dokument, kliknou na „Vyhodnotit" a aplikace odešle věty dokumentu do LLM k analýze. LLM vrátí strukturovaný JSON s informací, které věty porušují která pravidla.

V **pátek máme celodenní workshop s ~15 souběžnými uživateli**. Každé kliknutí na „Vyhodnotit" spustí **6 paralelních LLM volání** (jedno na jazykové pravidlo), z nichž každé obsahuje dávku vět z dokumentu uživatele. Typický dokument má 20–80 vět.

Potřebujeme **LLM hostovaný na ÚFAL**, servírovaný přes **vLLM** (continuous batching je nezbytný pro souběžné uživatele).

## Co jsme testovali

Provedli jsme zátěžové testy tří ÚFAL modelů na vLLM — měřili jsme rychlost odpovědi a kvalitu výstupu (korektnost strukturovaného JSON, sémantická přesnost oproti anotovaným referenčním datům).

| Model | Dávka 10 vět | Dávka 30 vět | 8 souběžných uživatelů (30 vět) | Kvalita |
|---|---|---|---|---|
| **Apertus 8B** (vLLM) | 13s, funguje | 28s se splitováním | 59s medián čekání | Použitelný — vysoký recall, ale hodně false positives |
| **EuroLLM 22B** (vLLM) | 57s, částečně | **Timeout** (120s) | Netestováno (příliš pomalé) | Slabý — vynechává věty, pomalý |
| **Apertus 70B FP8** (4×3090) | 98s | Netestováno (příliš pomalé) | Netestováno | Nejlepší kvalita, ale příliš pomalý |

**Hlavní zjištění:** Model 8B funguje, ale má problémy s kvalitou — označuje příliš mnoho false positives (přesnost ~25 %) a při delších dávkách vynechává věty. Vynechávání jsme zmírnili splitováním promptů (odesíláme ≤15 vět na volání), ale nízká přesnost přetrvává. Větší model by kvalitu výrazně zlepšil.

## Náš požadavek

Rádi bychom na pátek získali **dedikovanou vLLM instanci** s následujícími parametry:

1. **Model: Apertus 70B** (nebo srovnatelný model 22B+ s dobrou podporou češtiny)
2. **Servírování: vLLM** (ne Ollama — potřebujeme continuous batching pro souběžné uživatele)
3. **Hardware: dostatečně rychlý pro 70B s latencí <30s na požadavek**
   - Současný Apertus 70B na 4×RTX 3090 dává ~90s na požadavek — příliš pomalé
   - A100 80GB nebo H100 by to pravděpodobně stáhly pod 20s
   - Alternativně model 22B na současném hardware, pokud rychlejší GPU nejsou k dispozici
4. **Dostupnost: celý pátek, dedikovaně** (nesdíleno s jinými úlohami během workshopu)
5. **API: OpenAI-kompatibilní** endpoint na `ai.ufal.mff.cuni.cz` (stejné jako současné nastavení)

6. **API klíč:** Prosíme o dedikovaný servisní API klíč pro aplikaci SPRINT (pokud je současný klíč pouze testovací nebo osobní)

**Záložní varianta:** Pokud rychlý 70B setup není možný, použijeme Apertus 8B — funguje, jen s nižší kvalitou. Prosíme o potvrzení, že bude v pátek dostupný.

## Technické detaily (pro úplnost)

- API endpoint: `https://ai.ufal.mff.cuni.cz/api/chat/completions`
- Autentizace: Bearer token (máme funkční API klíč)
- Očekávaná zátěž: ~15 uživatelů, ~6 požadavků na vyhodnocení, dokumenty o 20–80 větách
- Se splitováním promptů: ~12–36 požadavků na jedno uživatelské vyhodnocení (chunky po ≤15 větách)
- Špičkový počet souběžných požadavků: ~50–100
- Parametry: `temperature=0`, `max_tokens=8192`
- Formát odpovědi: JSON pole, ~50–200 tokenů na větu v dávce
