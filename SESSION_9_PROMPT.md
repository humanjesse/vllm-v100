Continue the V4-Flash-on-V100 port. Session 8 hit first-runnable on
real V4-Flash with TP=8 — 16 finite tokens for "Hello" in 4.9 s, but
the output is GIBBERISH (`);); hihicnhibbcncncnukukiukiuki`). Per
`project_v4_flash_topk_sensitivity.md`, incoherent / repetitive output
is a WIRING BUG (not fp16-vs-bf16 topk-flip noise — that produces
plausible-but-different output, not nonsense). Session 9's primary job
is to find and fix that wiring bug.

Read the handoff docs first:

  - /home/admin/vllm-v100/SESSION_8_CONTINUATION.md (full session-8
    details: TP plumbing in V4Attention, attn_sink TP loader, wo_a
    quant-flip at n_local_groups==1, _DEQUANT_PATHS made tp-aware,
    compressor/indexer stay replicated, three engine-init knobs
    needed on top of TP plumbing — None spec + max_num_seqs=4 +
    enable_prefix_caching=False).
  - /home/admin/vllm-v100/SESSION_7_CONTINUATION.md (V4Attention.forward
    + DeepseekV4MoE.forward + decoder/model wiring at tp=1 — synthetic-
    config forward smoke passes; multi-step generation NOT exercised
    there because it's a single-prompt single-step test).
  - /home/admin/vllm-v100/SESSION_6_CONTINUATION.md (loader, dequant
    pre-processor, hash-MoE tid2eid).
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md
    (cumulative project state through session 8, open work).
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md
    (REQUIRED reading — explains why gibberish ≠ topk-flip).

Don't re-derive any of that. VERIFY before acting:

1. Run the 11-file overlay verification one-liner from
   SESSION_6_CONTINUATION.md ("Site-packages overlays" section). All
   eleven MUST report MATCH. (Only `model_executor/models/deepseek_v4.py`
   was edited in session 8.)
2. TileLang common.h SM70 patch is still applied (run from /tmp, NOT
   from the repo dir):
     cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
       -c "from vllm.model_executor.layers.deepseek_v4_v100_kernels import patch_tilelang_sm70; patch_tilelang_sm70(verbose=True)"
   Should print "already patched".
3. Working tree on branch v4-flash-v100 at /home/admin/vllm-v100. Memory
   says session 8 left 5 modified + 14 untracked files. Confirm with
   git status. Don't commit/push without asking — user wants to defer
   ALL commits until end-to-end working AND coherent.
4. Confirm regression tests (each as a script, NOT pytest):
     a) Session-4 backend test (~30s):
        cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
          /home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py
        Expect "ALL PASS".
     b) Session-5 instantiation test (<10s, updated for None spec):
        cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
          /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py
        Expect "ALL 5 PASS".
     c) Session-7 forward smoke (~10-15s, includes TileLang JIT):
        cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
          /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_forward_smoke.py
        Expect "ALL PASS".
     d) Session-8 TP=8 instantiation (CPU, 8 procs, ~60s):
        cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
          /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_instantiation.py
        Expect "8/8 ranks passed".
     e) Session-8 TP=8 serve smoke (~3 min cached):
        cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
          /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_serve_smoke.py
        Expect to print `[3/3] Output:` with `prompt = 'Hello'` and a
        16-token completion. Output will still be gibberish at session
        9 start — that's exactly what to fix.

Goal this session: Task #8 — DEBUG THE FORWARD WIRING under multi-step
generate so that V4-Flash produces COHERENT output. The bar is "outputs
look like English / look like training data", not exact-token-match
against the bf16 reference (use distribution-level metrics for that —
PPL or top-k overlap). This is intentionally narrow: no perf, no
multi-request scheduling, no vLLM-paged caches — just correctness.

## Likely culprits (priority order)

1. **Multi-step decode positions / kv_cache writes.** vLLM v1's runner
   calls `forward(positions, hidden_states)` once per generation step.
   For a prompt of length N and decoded-so-far K tokens, the K-th decode
   step gets `positions = [N+K-1]` and `hidden_states = [1, hidden]`.
   Our V4Attention.forward derives `start_pos = positions[0].item()` and
   asserts `seqlen == 1` when `start_pos > 0`. Then writes
   `self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`. Suspect:
   - Does vLLM actually pass cumulative position, or position relative
     to the request? Add `print(positions, start_pos)` at the top of
     forward and run the smoke test for one decode step. If positions
     don't increment by 1 each step, our kv_cache write pattern is
     wrong (we'll smear or miss writes).
   - Does the prefill step hand us `positions = [0, 1, ..., N-1]` (good)
     or something else (bad)?
   - Is `forward` actually called once per layer per step, or batched
     somehow? Side effects on `self.kv_cache` only work if forward is
     called per-step in order.
   The cleanest test: hook a `print` into V4Attention.forward (just one
   layer, layer_id==2 for ratio==4) and grep the smoke-test log for the
   actual `positions` pattern across steps.

2. **Compressor / Indexer state under multi-step.** The compressor's
   `kv_state`/`score_state` buffers accumulate across calls. At
   `start_pos = 0` we initialize fresh; at `start_pos > 0` we update
   incrementally. But across vLLM calls, are we starting fresh between
   requests? Currently NO — the buffers persist. For a single-request
   test this is OK, but if vLLM re-runs prefill (e.g. for a second
   prompt) without resetting, state will be stale. Verify by adding a
   debug print at the top of `V4Compressor.forward` showing
   `kv_state[:bsz, :ratio]` magnitude before vs after each call.

3. **`wo_a` quant-flip math at TP=8.** The "ColumnParallelLinear with
   non-replicated input" trick assumes (a) ColumnParallelLinear's
   weight shard at `tp_rank=r` is exactly the per-group block for group
   `r`, and (b) wq_b's column shard at rank r contains exactly group
   r's heads. (a) requires the checkpoint's `wo_a.qweight` to be laid
   out so that contiguous output rows correspond to contiguous groups
   — verify by reading wo_a.qweight from one shard and checking the
   column structure. (b) is true by ColumnParallelLinear convention.
   Quickest check: at tp=1, the smoke test's output should be coherent
   (the per-group einsum path is well-tested via the session-3
   numerical-equivalence test). Run the TP=1 serve smoke (need to write
   one — copy `test_deepseek_v4_v100_tp8_serve_smoke.py` and change
   `tensor_parallel_size=8` to `1` and `max_num_seqs=4` stays). If
   tp=1 also produces gibberish, the bug is NOT in wo_a TP path. If
   tp=1 is coherent and tp=8 is not, wo_a is the smoking gun.

4. **`attn_sink` TP loader slice direction.** The closure does
   `loaded_weight[tp_rank * n_local : (tp_rank+1) * n_local]`. Verify
   this aligns with how wq_b shards heads. ColumnParallelLinear shards
   along output dim with rank `r` getting `[r*n_local_heads,
   (r+1)*n_local_heads]`. attn_sink is `[n_heads]` per the checkpoint
   — so head `h` corresponds to attn_sink[h]. Rank r owns heads
   `[r*n_local, (r+1)*n_local]`, so it should own attn_sink[r*n_local
   : (r+1)*n_local]. ✓ Direction matches. But double-check: what's the
   actual head-to-rank mapping in vLLM's ColumnParallelLinear? It might
   shard via interleaving rather than contiguous slicing.

5. **Hash MoE input_ids stashing under multi-step.** For each forward
   call, `DeepseekV4MoE.forward` does `self._cached_input_ids =
   input_ids.flatten()`. For decode (single token), this is `[token_id]`
   of length 1. The `_v4_routing` then does
   `tid2eid[self._cached_input_ids]` which gives `[1, n_activated]`
   indices. That should be correct for single-token decode. Verify by
   logging `input_ids` and the resulting `indices` from one decode step.

6. **`process_weights_after_loading` for compressor/indexer.** Plain
   `nn.Linear` in our model class doesn't go through vLLM's
   process_weights_after_loading. The fp16/fp32 weight is just copied
   in via the dequant pre-processor + default loader. But the W4A16
   triple is stored on disk in `[K_packed, N]` layout (column-major-ish
   for output), and after dequant we transpose to `[N, K]` to match
   `nn.Linear.weight`. Check that the transpose direction is correct
   by comparing output of compressor.wkv on a fixed input vs the
   reference's bf16 output.

## Concrete debug plan (do this in order)

### Step 1 — Establish a baseline at tp=1

Write `tests/models/test_deepseek_v4_v100_tp1_serve_smoke.py` (copy of
the tp8 smoke, change `tensor_parallel_size=8` → `1`, drop the
`enable_prefix_caching=False` if possible — it should still work at
tp=1 because spec=None gives 0 groups regardless of tp). Run it. The
tp=1 model loads ALL weights on a single V100 — but V4-Flash is 143GB
and a V100 has 32GB. So tp=1 against the real V4-Flash CAN'T LOAD.

Alternative: build a small V4-shaped test config (4 layers, 4 experts,
hidden=128) like the session-7 forward smoke does, but RUN THROUGH
`vllm.LLM(...)` not through the synthetic test harness. This exercises
multi-step generate at tp=1 with random weights. If THIS produces
gibberish, the bug is generation/decode-loop, not TP. If it produces
"random but coherent-looking" garbage (uniform noise across vocab),
the bug is TP-specific. The bar at random init is low — output won't
make sense regardless — but the DISTRIBUTION matters.

### Step 2 — Instrument V4Attention.forward at one layer

Add temporary print statements in V4Attention.forward (gated by
`if self.layer_id == 2 and torch.cuda.current_device() == 0:`):

  - `start_pos`, `positions.shape`, `seqlen`
  - `q.abs().mean()`, `kv.abs().mean()`, `o.abs().mean()` after sparse_attn
  - For each decode step, print `self.kv_cache[:bsz, :start_pos+1].abs().mean()`
    — should grow as more tokens are written.

Re-run the tp8 smoke. Look for: positions incrementing by 1 each step,
kv_cache mean growing monotonically, q/kv/o magnitudes staying in fp16
range (no NaN/Inf, no all-zero).

### Step 3 — Add reference comparison for ONE decode step

Use `/tmp/v4flash/inference/generate.py` to generate the same "Hello"
prompt with the bf16 reference (single-rank, won't fit on V100 but you
can ATTEMPT to run the reference on CPU via `device='cpu'` to dump the
expected logits for the first decode step). Then compare against the
tp8 V100 logits. If reference logits look like a plausible distribution
(top-1 token is something like ", ", " world", a newline, etc.) but
ours are uniform / concentrated on garbage tokens, our forward is
broken. The session-3 numerical-equivalence test framework is the
right pattern.

### Step 4 — Bisect

Once you have a smoking gun (e.g. "kv_cache magnitude is wrong after
step 5"), bisect by selectively reverting parts of session-8 changes:
- Revert `attn_sink` to full-size `[n_heads]` and slice in forward
  (instead of TP-slicing at load time). If output becomes coherent,
  attn_sink TP loader is buggy.
- Revert `wo_a` to plain nn.Linear+einsum even at TP=8 (just remove
  the `_wo_a_quant` flag to force False). Add `.attn.wo_a.` back to
  `_DEQUANT_PATHS`. If output becomes coherent, wo_a quant flip is
  buggy.
- Run TP=4 (n_local_groups=2 → uses the per-group einsum path, NOT
  the wo_a quant flip). If TP=4 is coherent, wo_a is the bug.

## Smallest viable milestone

Test prompt "Hello, my name is" and check that the completion is at
least `english_words` (whatever they are), not `cncncnukukiukiuki`.
Don't worry about correctness vs reference yet — just coherence.

## Constraints to respect (durable)

- Don't merge to main / push to origin without asking. User has
  explicitly asked to defer ALL commits until end-to-end working AND
  coherent.
- Don't download V4-Flash again (already at /home/admin/models/V4-Flash-W4A16/).
- Existing fork patches (TurboMindAsymLinearKernel, qwen3_5 patches,
  awq_sm70_moe, flash_attn_v100) MUST keep working.
- vllm in venv is wheel-installed (NOT editable). Every file edit must
  be cp-overlaid to site-packages.
- supports_compute_capability MUST stay strict-V100 (cc==(7,0) only).
- supported_dtypes = [torch.float16] only.
- If you see "high rel_err" against the reference (i.e. plausible but
  different), check the topk-flip memory FIRST. Gibberish is a wiring
  bug, NOT topk-flip.

## Specific landmines (durable from earlier sessions)

- **wo_a quant flip is a lockstep change.** If you make wo_a a quant
  Linear, you MUST remove `.attn.wo_a.` from `_DEQUANT_PATHS_BASE` /
  `_dequant_paths()` at the same time. Otherwise the loader dequantizes
  and writes into a `.qweight` slot that no longer exists → silent
  miss into `_unmatched_weights`.
- **Misleading manifest:** `model.safetensors.index.json` lists
  embed.qweight/qzeros/scales but the actual shard has embed.weight.
- **g_idx is a no-op for sym=True**, synthesized as "loaded by initialization".
- **session-2 V4Compressor.wkv is fp32 nn.Linear**, with the reference
  casting `x.float()` before applying. Don't change.
- **scoring_func="sqrtsoftplus"** wired via `custom_routing_function=self._v4_routing`.
- **TileLang JIT cost:** first sparse_attn call at a new
  (h, d, m, n, topk) signature triggers a compile. ~10s per signature.
- **set_default_torch_dtype is the silent-killer category.** When
  constructing the model outside the production loader, wrap in
  `set_default_torch_dtype(torch.float16)`.
- **process_weights_after_loading + init_workspace_manager + set_forward_context**
  are required for FusedMoE forward outside the production path.
- **Three engine-init knobs** required for V4-Flash TP=8 (per session 8):
  None spec, max_num_seqs=4, enable_prefix_caching=False. Don't
  remove these without an alternative.

## Update at session end

- Overlay table in SESSION_8_CONTINUATION.md (or write SESSION_9_CONTINUATION.md).
- Auto-memory `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
  with session-9 progress.
- MEMORY.md index entry.

Auto mode is fine. Coherent output is the bar; partial progress is OK
(this is a debug session, not an implementation session). Be honest
about what you ruled out vs what you confirmed.
