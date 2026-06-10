# Pace landing page — pattern mine + draft

## Part 1 — Pattern mine

### 1. CodeVetter — the template (https://codevetter.com)

**Hero**: "Stop merging unreviewed AI code." (5 words, imperative, names an enemy
action). Sub-head: "CodeVetter is a desktop review cockpit for the diffs your
agent ships. Catch what Cursor, Claude Code, and Devin missed — vulnerabilities,
regressions, and silent drift. Runs entirely offline." (32 words, names
competitors by name, ends on the moat — "entirely offline"). Dual CTA:
**"Download for macOS"** + **"See it in action"**. No email gate, no waitlist.

**Trust strip directly under the hero** (single line, bullet-separated):
"No telemetry · Bring your own key · Open source · ISC · Signed binaries."
This is the highest-leverage element — it pre-empts every objection in one row.

**Hero visual**: a desktop app mockup showing the actual review cockpit —
pipeline (diff parse → AST trace → LLM score), findings list with severity tags,
a real code snippet flagged for SQL injection with a suggested patch. Not a
video; a dense product screenshot that proves the thing exists and is real.

**Body order**:
1. "What the review looks for" — concrete list of 14 CWE/OWASP classes
2. "Built for the way agents ship code" (feature grid)
3. Six sub-feature blocks, each one verb-led headline: "Diff-aware review
   engine", "Bring your own key", "Runs offline", "Git-native",
   "Severity-tiered", "Patch suggestions, not pep talks"
4. "Operating loop" — 3-step workflow
5. "Provider matrix" — concrete table (Anthropic / OpenAI / OpenRouter / local)
6. Pricing — $0 / $12 / Custom, three tiers, no gimmicks
7. Final CTA + footer

**Why it works**: every section is verifiable, named enemies in the hero, the
trust strip neutralizes objections before they form, the screenshot is the
product (not a hero illustration), feature headlines are all verb-led and
concrete. ~1,200 total words; nothing decorative.

### 2. Wispr Flow — https://wisprflow.ai

Hero: "Don't type, just speak." (4 words, imperative). Sub: "The voice-to-text
AI that turns speech into clear, polished writing in every app." (14 words).
CTA: "Download for free". Hero visual = static before/after text comparison.
Mid-page logo carousel (Vercel, Notion, Amazon, Nvidia, ~20 logos). 9
testimonials scattered. 4-column feature row anchored by the "4x faster
(220 wpm vs 45 wpm)" claim. **Over-claims to counter**: "Privacy & Security:
your data, your control" plus SOC2/HIPAA badges, but the product still
round-trips your audio to their cloud. Their privacy story is policy-based,
not architecture-based. Pace's counter: "Your audio never leaves the Mac.
Not a policy — an architecture."

### 3. Dottie — https://dottie.ai (403 on direct fetch; positioning via search)

Headline pattern: "Your Private Voice Assistant for Mac." Positioning:
"controls your Mac through voice", "134 system tools", "3,800+ local models
via Ollama/MLX", "free forever". They are the closest competitor and they own
the "private + Mac + voice + free" quadrant. **What to undercut**: (a) they
proxy through Ollama/MLX which runs on GPU, not ANE — Pace is the only one on
the chip Apple built for it; (b) they're a generalist shell over a single
local model — Pace ships a specialist LoRA per skill; (c) "3,800 models" is
breadth-as-vanity — Pace's pitch is one model per skill, tuned, not 3,800
untuned.

### 4. Cluely — https://cluely.com

Hero: "#1 Undetectable AI for Meetings" (5 words, audacious + ranked).
Sub: "Cluely takes perfect meeting notes and gives real-time answers, all
while completely undetectable" (15 words). Dual CTA (App Store + desktop).
Static mockup hero. No pricing on page. **Viral mechanic**: the tone IS the
product — one edgy claim, repeated. "Undetectable" is morally loaded and
that's the engine. **Lesson for Pace**: one sentence has to do the work. Pace's
equivalent edge claim is "runs on the chip Apple built for it" — don't soften
it. Also: Cluely is the breach risk story we own ("sees your screen without
sending a pixel anywhere").

### 5. Linear — https://linear.app

Hero: "The product development system for teams and agents" (9 words; the
"and agents" is the 2026 positioning shift). Sub-head: "Purpose-built for
planning and building products. Designed for the AI era." (12 words). CTA:
"Issue tracking is dead" — provocative, links to a feature page. Hero is a
static carousel of 3 dense product screenshots (no video). Social proof
placement: lower on page with customer quotes from OpenAI, Ramp, Opendoor,
and the metric "Linear powers over 33,000 product teams." Body is **5
narrative modules** labelled FIG 1.0 → 5.0 (Intake / Plan / Build / Diffs /
Monitor) — single-column, sequential, each with a real working example
(ENG-2703 with agent "Codex"). No pricing on page, no FAQ. Footer is 6
columns. **Gold standard moves**: hyperspecific product language, real
issue numbers in screenshots (not "Lorem ipsum"), agent-as-collaborator
framing not automation theatre, narrative arc not feature dump.

---

## Part 2 — Pace landing draft

> Template: CodeVetter (https://codevetter.com). Hero shape + trust-strip
> + dense product screenshot + verb-led feature blocks + concrete numbers +
> no email gate. Linear's narrative-module body where it fits.

### Nav
Pace · Features · How it works · Skills · FAQ · Download

### Hero
**Stop sending your voice to someone else's cloud.**

Pace is a voice-first Mac assistant where every skill has its own brain.
100% on-device. ANE-accelerated. Sees your screen and types in your apps —
without sending a single pixel anywhere.

**[Download for macOS]**   **[Watch the 12-second demo]**

### Trust strip (single line, directly under hero)
No cloud · No Private Cloud Compute · No Gemini · No telemetry · Signed by Apple Developer ID · English + Mac only, by design

### Hero visual (placeholder)
12-second muted autoplay loop, single take, no cuts. Push-to-talk fires →
waveform pulses → Mail compose window opens at 159ms → email body streams
into the field word-by-word → TTS reads the subject line out loud. Bottom
overlay timer: `119ms TTFW · 491ms first audible word · 17 tok/s on ANE`.
Loops cleanly.

### By the numbers (strip, Linear-style placement)
- **119 ms** — time to first planner token (warm)
- **159 ms** — Mail compose window visible
- **491 ms** — first audible word from TTS
- **17 tok/s** — ANE inference today, **30+** targeted via macOS 26 int8 handoff
- **1.5 GB** — total bundle (WhisperKit + Qwen3-base + every adapter)
- **0** — bytes sent to any server, ever

### Feature blocks (verb-led, CodeVetter-style)

**Every skill has its own brain.**
Pace doesn't run one generalist model and pray. Each skill ships its own
LoRA: a planner (Qwen3-0.6B, 70% on compose, 33% on action-routing), an
embedder (Qwen3-Embedding-0.6B), a vision model (UI-Venus 1.5–2B port for
screen understanding), and ASR (WhisperKit large-v3-turbo). One model
per job, tuned for that job. The opposite of Ollama-as-a-shell.

**Runs on the chip Apple built for it.**
Pace is the only Mac assistant that targets the Apple Neural Engine, not
the GPU. Dottie, Ollama, MLX-LM — all GPU. The ANE is the chip Apple
designed for exactly this workload, and Pace is what it's for.

**Voice in, voice out, JARVIS-class.**
Push-to-talk. Streaming TTS so the assistant starts speaking before it
finishes thinking. Voice editing — interrupt mid-sentence and revise.
First audible word in 491 ms warm.

**Sees your screen. Types in your apps. Sends nothing.**
Vision OCR runs locally via Apple's Vision framework. UI dispatch goes
through AX, EventKit, MessageUI, Shortcuts. The screen never leaves the
Mac. The pixels never leave the Mac. There is no upload path in the
binary — not by promise, by architecture.

**Apple-native, not Electron-native.**
AX dispatch, EventKit, MessageUI, Shortcuts CLI, Vision OCR. mailto://
opens Mail in 159 ms because we don't fight the OS, we ride it. Signed
DMG, notarized, native macOS 26.

### How it works (3 steps, CodeVetter-style operating loop)
1. **Hold the hotkey.** Speak. WhisperKit transcribes on-device.
2. **The planner routes.** Picks the right skill LoRA, picks the right
   action (compose / search / open / dispatch), picks the right app.
3. **Pace acts and speaks.** Streams text into your app while streaming
   speech to your speakers. You can interrupt and revise mid-stream.

### Skills shipping in v1
Compose mail · Reply mail · Calendar add · Calendar query · Reminders ·
Notes capture · Open app · Screen read · Spotlight intent · Shortcuts
dispatch · Voice edit · Voice search.

### Counter-positioning (no header — woven into FAQ to avoid sounding bitter)

### FAQ

**Is this really 100% local?**
Yes — by architecture, not by policy. The binary has no network egress
for inference. You can pull the cable.

**Apple is shipping Gemini in Siri at WWDC 2026. Why use Pace?**
Because Private Cloud Compute is still cloud, and Gemini-in-Siri is still
Google touching your prompts. Pace is the version where the model lives
on your Mac.

**How is this different from Dottie?**
Dottie is a shell over Ollama / MLX on the GPU, with one generalist
model. Pace is a per-skill LoRA stack on the ANE. Dottie's pitch is
"3,800 models you could load"; Pace's pitch is "the right model already
loaded for what you just said."

**How is this different from Wispr Flow?**
Wispr round-trips your audio to its cloud. Pace doesn't.

**How is this different from Cluely?**
Cluely sends your screen to their servers. Pace sees your screen locally,
acts locally, sends nothing. Cluely had a breach. Pace cannot have one of
the same shape because the data is never in transit.

**Does it work offline?**
Yes. On a plane, in a SCIF, on a desert island — same latency.

**What Macs?**
Apple Silicon, macOS 26+. M1 and up. English-language only in v1.

**Price?**
Free in v1. No account. No telemetry. No email required.

**Who builds it?**
Solo — Sarthak Agrawal ([sarthakagrawal.dev](https://sarthakagrawal.dev)).

### Final CTA
**Download Pace for macOS** — 1.5 GB DMG, signed, notarized, no account.
[Download] [GitHub]

### Footer
Pace · Built on a Mac, for a Mac · No cloud, no account, no telemetry
Product: Features · Skills · How it works · Download
Project: GitHub · Changelog · Roadmap
Legal: Privacy (there is nothing to take) · License · Contact
© 2026 Sarthak Agrawal

---

## Notes for the writer

- The hero replaces the candidate headline. The candidate ("A voice
  assistant that lives on your Mac — not in someone else's cloud") was 15
  words and conceded ground by naming the cloud. CodeVetter's "Stop X"
  imperative is sharper. Pace's "Stop sending your voice to someone
  else's cloud" is 8 words and names the enemy action.
- Position #4 ("runs on the chip Apple built for it") is the sharpest line
  and has been promoted to its own feature block headline.
- The trust strip under the hero does the same job CodeVetter's does:
  pre-empts every "but is it really" objection in one row.
- The "by the numbers" strip uses Linear's placement pattern; the 0-bytes
  number is the moat.
- The 12-second demo placeholder has a specified storyboard, not "insert
  video here."
- No pricing table — there is one price (free). Stating it in the FAQ is
  enough; a table would be theatre.
