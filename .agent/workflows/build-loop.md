# build-loop.md
# Save this to: .agent/workflows/build-loop.md in the project root
# Trigger in Antigravity chat with: /build-loop

You are working inside the inference-serving project. Follow this loop
exactly — do not skip steps or merge them to save time.

1. **Draft the task list.** If one doesn't already exist for this phase of
   PRD.md, generate a Task List artifact breaking the current goal into
   small, independently verifiable units. Each unit should be completable
   and testable in isolation.

2. **Load context for one task only.** Pull in only the files, skills, and
   PRD sections relevant to the single next task. Do not pre-load context
   for later tasks — this is what keeps long sessions from degrading.

3. **Implement.** Write the code for that one task.

4. **Verify before reporting done.** Run the test suite and linter for the
   affected area. If this task touches the serving path (model loading,
   batching, quantization, autoscaling), run the relevant benchmark and
   compare against the last recorded baseline in `benchmarks/results/`.
   A task is not complete until this step passes.

5. **Checkpoint.** Report: what was built, what was verified, what the
   benchmark delta was (if applicable), and what's next. Stop here and
   wait for explicit approval.

6. **Loop.** On approval, clear working context and return to step 2 for
   the next task. Do not carry implementation detail from the completed
   task forward — only the fact that it's done.

If a task turns out to need something outside the current plan (new
dependency, infra change, PRD ambiguity), stop immediately and surface it
instead of guessing and continuing.
