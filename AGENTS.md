# AGENTS.md

## Project
Self-hosted LLM inference-serving platform: quantized model serving (vLLM/TGI),
Kubernetes deployment, autoscaling under load, and a benchmark suite comparing
latency/throughput/cost against a naive HF Transformers baseline.
Full scope is defined in `PRD.md` — read it before starting any task.

## Non-negotiable process
1. **Plan before touching code.** Every task starts with an Implementation Plan
   artifact. Do not begin editing until the plan has been explicitly approved.
2. **Read the skill before executing.** Check `H:\Projects\Cinch\.agent\skills` for anything
   relevant to the current task before writing code — do not reinvent a
   pattern that's already documented there.
3. **Stop-and-report checkpoints.** After each unit of work (one task from the
   task list), stop, summarize what changed and why, show test/lint output,
   and wait for review before starting the next task. Do not chain multiple
   tasks together without a checkpoint in between.
4. **No silent scope expansion.** If a task reveals a need for something not
   in the current plan (a new dependency, a schema change, a new service),
   stop and flag it instead of absorbing it into the current task.

## Testing bar
No task is "done" without tests. Target: every serving-path change ships with
at least one test that would have caught the bug it fixes or the regression
it risks. Treat ARES's standard (91%+ coverage, tests as the definition of
done, not an afterthought) as the bar for this repo too.

## Git conventions
- Conventional commits (`feat:`, `fix:`, `bench:`, `docs:`, `chore:`)
- No direct commits to `main` — feature branch + review, even solo
- Every PR description states: what changed, why, and the benchmark delta
  if the change touches the serving path

## Safety guardrails
- Never run a destructive command (`rm -rf`, force-push, `kubectl delete`
  against a non-local context) without explicit approval first
- Never commit API keys, cloud credentials, or `.env` contents
- Any change to autoscaling policy or GPU resource requests gets called out
  explicitly before applying — these have real cost implications

## Anti-slop standard
No filler README sections, no unsubstantiated "blazing fast" claims. Every
performance claim in docs must trace to a number in `benchmarks/results/`.
