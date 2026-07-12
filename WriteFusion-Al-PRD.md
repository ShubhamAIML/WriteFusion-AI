# Product Requirements Document: WriteFusion AI

**Author:** Shubham | **Repo:** [ShubhamAIML/WriteFusion-AI](https://github.com/ShubhamAIML/WriteFusion-AI)
**Status:** Live (deployed) | **Version:** 1.0
**Date:** July 2026

---

## 1. Overview

WriteFusion AI is a Flask + TensorFlow web application that serves a locally-trained Keras Transformer language model for next-word prediction, story/paragraph continuation, and model diagnostics. It exposes a polished browser UI so users can interact with a from-scratch language model — type text, see live word suggestions with probabilities, generate continuations under multiple decoding strategies, and inspect model internals like attention and perplexity.

## 2. Problem Statement

Most people who use "AI text prediction" only ever see it as a black-box feature inside a product (phone keyboard, Gmail Smart Compose). There's no easy way to:
- See *how* a Transformer actually predicts the next word (ranked candidates, probabilities, attention).
- Compare different decoding/sampling strategies side-by-side and understand their effect on repetition, creativity, and coherence.
- Interact with a fully self-trained (not fine-tuned/third-party) language model in a transparent, inspectable way.

WriteFusion AI solves this by pairing a custom-trained Transformer with a UI designed for both casual use (writing assistant) and technical exploration (model diagnostics).

## 3. Goals & Non-Goals

**Goals**
- Provide real-time next-word prediction with ranked, probability-scored suggestions.
- Support long-form story/paragraph generation with configurable decoding strategies.
- Make model internals (attention focus, perplexity, architecture) visible to the user.
- Run reliably on a free-tier deployment (Render) despite TensorFlow's resource footprint.
- Demonstrate end-to-end ML engineering: data pipeline → training → serving → UI, as a portfolio-grade project.

**Non-Goals**
- Not a production-scale writing assistant (no multi-user accounts, no persistence of user documents).
- Not aiming to match commercial LLMs (GPT-class) in generation quality — the value is transparency and from-scratch engineering, not state-of-the-art output.
- No mobile app; web-responsive only.

## 4. Target Users

- **Primary:** Recruiters/hiring managers and ML engineers evaluating Shubham's applied ML skills (portfolio use case).
- **Secondary:** Students/developers curious about how Transformer decoding strategies (Top-K, Top-P, Typical, Min-P, etc.) affect generated text.
- **Tertiary:** Casual users wanting a lightweight next-word/story-completion writing aid.

## 5. Core Features (Current)

| Feature | Description |
|---|---|
| Live next-word prediction | Predicts next word as the user types; shows ranked candidates with probabilities |
| Suggestion acceptance | Tab (desktop) / Enter (mobile) to accept top suggestion |
| Story & paragraph generation | Extends user text into longer continuations in the same editor |
| Multiple decoding strategies | Greedy, Temperature, Top-K, Top-P, Top-K+Top-P, Typical, Min-P, Smart Anti-Repetition |
| Anti-repetition controls | Repetition penalty, no-repeat n-gram blocking, dynamic temperature/Top-K |
| Attention inspection | Displays attention-focus tokens from the Transformer |
| Perplexity probe | Computes perplexity of supplied text for confidence/quality checking |
| Model diagnostics popup | Shows architecture, training pipeline, and hyperparameters |
| Light/dark, responsive UI | Works across desktop and mobile |

## 6. Technical Architecture

- **Backend:** Flask, serving a Keras/TensorFlow Transformer model (`best_model.keras`) with saved tokenizer (`tokenizer.pkl`) and config (`model_config.json`).
- **Frontend:** Single-page UI (`templates/index.html`) with light/dark theming.
- **Corpus:** ~2.7M words / 860,173 sentences used for training.
- **Deployment:** Render (free tier), configured via `render.yaml`, `runtime.txt`, `requirements.txt` (TensorFlow ≥2.21.0, protobuf pinned to 6.31–8.0 range to avoid runtime mismatch errors).
- **API routes:**
  | Route | Method | Purpose |
  |---|---|---|
  | `/` | GET | Loads main UI |
  | `/api/model-info` | GET | Model metadata (params, vocab size, layers, heads, context length, etc.) |
  | `/api/complete` | POST | Next-word candidates + attention data |
  | `/api/generate` | POST | Story/paragraph continuation |
  | `/api/perplexity` | POST | Perplexity score for supplied text |

## 7. Known Constraints

- Render free tier has limited CPU/memory; TensorFlow thread contention was an active issue that had to be resolved for stable serving.
- No GPU in deployment — inference is CPU-bound, capping generation length (max 500 words) and latency.
- Single-instance, stateless: no user accounts, no saved sessions/history.

## 8. Success Metrics (Suggested)

- Model serves predictions with acceptable latency (<2–3s) on Render free tier without crashing/restarting.
- Zero unhandled errors on `/api/complete`, `/api/generate`, `/api/perplexity` under normal load.
- Positive qualitative feedback from portfolio reviewers on UI polish and transparency of model internals.
- (Optional, if pursued) GitHub stars/forks or resume/interview conversion as an informal signal.

## 9. Future Enhancements (Proposed)

- Add basic usage analytics (requests per endpoint, latency) to demonstrate monitoring skills.
- Session-based history so users can revisit previous generations.
- Model card / versioning if retrained on larger corpus.
- Optional GPU-backed deployment tier for faster generation and longer max length.
- Export generated text (copy/download) directly from UI.

---

*This PRD was reconstructed from the public repository README and structure; sections 8–9 are suggested additions rather than confirmed existing functionality — flag any inaccuracies and I'll revise.*
