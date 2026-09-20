# Concurrent IFEval scheduling review

Read-only review; no GPU execution or rerun. PASS for task-output mutual exclusion in `run.py`.

The task-local `.running.lock` is acquired via nonblocking exclusive flock before selecting a generation-attempt filename, importing model/evaluation code, writing settings, or loading a GPU model. The local file handle stays alive throughout `main`. A competing worker returns immediately without mutating task outputs. The completed-results check occurs inside the lock, preserving an already finished result. Process exit releases the lock without a persistent stale-lock failure mode.

`schedule.py --tasks ifeval` replaces the default task sequence, so the new workers execute only IFEval. Qwen IFEval's effective batch size is uniformly 64 in the current adapter; other generation remains 16. The superseded BF16 partial batch-16 attempt is preserved and the current `settings.json` generation-log path identifies the intended attempt. Final auditing still needs to confirm that all three completed Qwen IFEval settings specify 64 and that the selected logs each have 541 samples.

The initially reported launcher-log race is resolved in the reviewed update: the scheduler reserves `run-N.log` with `open('x')`, increments N on FileExistsError, and exclusively creates `launch-N.json` as well. Competing schedulers therefore cannot overwrite the same reserved attempt log. This change was checked in source only, without a simulation or benchmark rerun. The ordinary exclusive-create reservation complements the child task lock; neither changes evaluation semantics.
