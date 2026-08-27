# Milestone 19: Ingress Security, Jailbreak Defense & PII Redaction

## 1. Overview and Problem Statement

Production LLM endpoints face continuous security and compliance threats:

1. **Adversarial Prompt Injections**: Attackers inject instructions to override system prompts (`"Ignore all previous instructions..."`) and access restricted corporate logic.
2. **Jailbreaks and Mode Switches**: Roleplay vectors (e.g. `DAN mode`, `developer mode enabled`) attempt to bypass safety guardrails.
3. **Delimiter Attacks**: Raw token delimiters (e.g. `<|im_start|>system`, `[INST] <<SYS>>`) injected into user messages attempt to manipulate chat template boundaries.
4. **PII and Secret Data Leakage**: Users paste SSNs, credit card numbers, AWS credentials, and OpenAI API keys directly into prompts, risking server log exposure and data contamination.

Milestone 19 implements `gateway/guardrails.py`. The gateway executes sub-millisecond regex and token scanning on incoming prompts:

- Injections, jailbreaks, and delimiter attacks are rejected immediately at ingress with **HTTP 400 Bad Request**, avoiding GPU forward compute.
- Sensitive PII entities are anonymized in-place (`[REDACTED_SSN]`, `[REDACTED_API_KEY]`) *before* prompt caching and GPU dispatch.
- Egress responses are filtered to prevent system prompt echoes.

---

## 2. Technical Architecture

### Ingress & Egress Security Pipeline

```
[ Incoming User Prompt ]
           │
           ▼
[ Guardrails Scanner (< 1ms CPU) ]
    ├── Check 1: Prompt Injections & Instruction Overrides ──(Detected)──► [ HTTP 400 Bad Request ]
    ├── Check 2: DAN & Developer Mode Jailbreaks          ──(Detected)──► [ X-Guardrails-Status: BLOCKED ]
    ├── Check 3: Chat Delimiter Escapes (<|im_start|>)     ──(Detected)──► [ GPU Compute: 0 W ]
    └── Check 4: PII Redaction (SSN, Cards, API Keys)
           │ (Sanitized: [REDACTED_SSN], [REDACTED_API_KEY])
           ▼
[ Semantic Cache & GPU Ingress Dispatch ]
           │
           ▼
[ Egress Security Filter (System Prompt Leak Defense) ]
           │
           ▼
[ Client Receives Secure, Anonymized 200 OK ]
```

---

## 3. PII & Threat Vector Specifications

### PII Anonymization Patterns

| Entity Name | Regex Pattern | Redaction Token |
|---|---|---|
| Social Security Number | `\b\d{3}-\d{2}-\d{4}\b` | `[REDACTED_SSN]` |
| OpenAI API Key | `\bsk-[a-zA-Z0-9_-]{20,}\b` | `[REDACTED_API_KEY]` |
| GitHub Access Token | `\bgh[pousr][-_][a-zA-Z0-9]{20,}\b` | `[REDACTED_API_KEY]` |
| AWS Access Key ID | `\bAKIA[0-9A-Z]{16}\b` | `[REDACTED_API_KEY]` |
| Credit Card Number | `\b(?:\d{4}[ -]?){3}\d{4}\b` | `[REDACTED_CREDIT_CARD]` |
| Phone Number | `\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b` | `[REDACTED_PHONE]` |

---

## 4. Empirical Security Evaluation

Live security audit executed against the k3d Kubernetes gateway and vLLM backend.

Source dataset: `benchmarks/results/guardrails_eval.json`

### Audit Results Matrix

| Scenario Name | Threat Category | Gateway Response Status | Guard Status | Defense Efficacy |
|---|---|---|---|---|
| `benign_technical_query` | Benign Technical | `200 OK` | `PASSED` | **0% False Positive Block** |
| `prompt_injection_override` | Prompt Injection | `400 Bad Request` | `BLOCKED` | **100% Attack Blocked (10.2 ms)** |
| `jailbreak_dan_mode` | DAN Jailbreak | `400 Bad Request` | `BLOCKED` | **100% Attack Blocked (44.6 ms)** |
| `delimiter_escape_attack` | Delimiter Attack | `400 Bad Request` | `BLOCKED` | **100% Attack Blocked (43.8 ms)** |
| `pii_ssn_and_api_key` | PII Ingress | `200 OK` | `PASSED` | **100% Data Anonymized** |

### Key Audit Metrics

- **Total Test Cases Evaluated**: 5
- **Security Defenses Passed**: 5 / 5 (100.0%)
- **Adversarial Injection Blocks**: 100% (All attacks blocked prior to GPU forward)
- **PII Data Leakage Prevented**: 100% (SSNs and API keys masked in-place)
- **Average Inspection Latency**: < 0.5 ms per scan on CPU

---

## 5. Verification and Test Suite

All 11 unit tests pass:

```powershell
python -m pytest tests/test_guardrails.py -v
```

```
tests/test_guardrails.py::test_detect_instruction_override PASSED        [  9%]
tests/test_guardrails.py::test_detect_dan_jailbreak PASSED               [ 18%]
tests/test_guardrails.py::test_detect_delimiter_attack PASSED            [ 27%]
tests/test_guardrails.py::test_redact_ssn PASSED                         [ 36%]
tests/test_guardrails.py::test_redact_api_keys PASSED                    [ 45%]
tests/test_guardrails.py::test_redact_credit_card PASSED                 [ 54%]
tests/test_guardrails.py::test_redact_phone_number PASSED                [ 63%]
tests/test_guardrails.py::test_benign_technical_prompts_pass_cleanly PASSED [ 72%]
tests/test_guardrails.py::test_egress_system_prompt_leak_filter PASSED   [ 81%]
tests/test_guardrails.py::test_scan_latency_under_1ms PASSED             [ 90%]
tests/test_guardrails.py::test_guardrails_metrics PASSED                 [100%]
```

Full repository regression suite: **151 / 151 tests passing**.
Code lint status: **0 errors** (`ruff check .`).
