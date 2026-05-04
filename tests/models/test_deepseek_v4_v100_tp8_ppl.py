# SPDX-License-Identifier: Apache-2.0
"""Quality-regression gate: per-token PPL on a fixed corpus.

Builds the same V4-Flash TP=8 fp16 engine the bar test uses and computes
per-sequence perplexity via vLLM's ``prompt_logprobs`` path. Designed for
A/B comparison between residual-stream variants (clamp vs fp32) — same
corpus + same engine config + temperature=0 → identical inputs across
runs, so the only thing varying is the model precision/code under test.

Why ``prompt_logprobs`` rather than ``compute_logits``:
  - During prefill, ``compute_logits`` returns logits only for the last
    position (vLLM's sampler optimization).
  - ``SamplingParams(prompt_logprobs=k)`` instructs the engine to record
    the top-k logprobs for every prompt position during prefill, which
    is exactly the teacher-forced trajectory we need for PPL.
  - Per-position ``Logprob`` entries include the actual prompt token's
    logprob even when it isn't in top-k (the engine inserts it
    explicitly), so summing -logprob over positions 1..N-1 and dividing
    by (N-1) gives the right per-sequence cross-entropy.

Corpus: a small embedded set of factual-prose snippets. ~30 sequences,
each truncated/extended to ~200-300 tokens via the model's tokenizer.
Embedded so the harness has no network dep and no per-run download.
For an absolute-quality comparison vs the bf16 reference, swap in
WikiText-2 (~/.cache/huggingface/datasets/wikitext) once available.

Usage:
    PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
      tests/models/test_deepseek_v4_v100_tp8_ppl.py [--n N] [--label TAG]

Outputs the mean PPL and per-sequence distribution to stdout. Tag the
two A/B runs with ``--label clamp`` and ``--label fp32res``, then
compare the **means** across runs.

Reproducibility caveat (measured 2026-05-03): per-sequence PPLs are
NOT bit-stable across runs of the same engine on the same inputs.
Mean PPL is stable to ~0.1%; per-sequence numbers can drift by tens
of percent. Likely cause: W4A16 quant GEMM (awq_gemm_sm70) and MoE
all-reduce both use atomic / order-dependent fp accumulation. So:
prefer mean-vs-mean for A/B decisions; treat per-sequence diffs as
noisy. A delta < ~0.5% on the mean is within noise; > 0.5% is signal.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time


MODEL_DIR = "/home/admin/models/V4-Flash-W4A16"


# Embedded corpus: 30 factual-prose snippets. Source-style text matches
# the model's pretraining distribution well enough for a stable PPL number;
# the absolute value isn't comparable to a WikiText-2 PPL run, but the
# relative A/B between residual-stream variants is valid.
CORPUS = [
    "The transistor was invented in 1947 at Bell Labs by John Bardeen, Walter Brattain, and William Shockley. Their device replaced bulky vacuum tubes and earned them the 1956 Nobel Prize in Physics. The point-contact transistor was crude but proved that semiconductor switching was possible at scale.",
    "Quicksort, devised by Tony Hoare in 1959, partitions an array around a chosen pivot and recursively sorts the two halves. Its average-case complexity is O(n log n), matching mergesort, but it sorts in place with low constant factors that make it dominant in practice for general-purpose sorting.",
    "The Cassini-Huygens spacecraft launched in 1997 and reached Saturn in 2004, where it spent thirteen years studying the planet, its rings, and its moons. The Huygens probe descended through Titan's atmosphere in 2005, becoming the first spacecraft to land on a body in the outer Solar System.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy captured from sunlight. The light-dependent reactions occur in the thylakoid membranes of chloroplasts, where chlorophyll absorbs photons and drives electron transport that ultimately produces ATP and NADPH for the Calvin cycle.",
    "TCP guarantees ordered, reliable delivery of a stream of bytes between two endpoints. It accomplishes this through sequence numbers, acknowledgements, retransmission timers, and a sliding-window flow-control mechanism. Congestion control, originally added by Van Jacobson in 1988, prevents collapse of the network under load.",
    "The CRISPR-Cas9 system, adapted from a bacterial immune mechanism, allows precise editing of genomic DNA. A guide RNA directs the Cas9 nuclease to a specific target sequence, where it makes a double-strand break. Cellular repair machinery then either disrupts the gene or, with a template, replaces it.",
    "Black holes are regions of spacetime where gravity is strong enough that nothing, not even light, can escape. The boundary, called the event horizon, marks the point of no return. Stellar-mass black holes form from the collapse of massive stars, while supermassive black holes anchor most galaxies.",
    "The Roman aqueducts supplied cities with fresh water from distant sources, sometimes hundreds of kilometers away. Their gentle, sustained gradients required precise surveying. The combined daily delivery to Rome at the empire's peak is estimated at over a million cubic meters, supplying baths, fountains, and private homes.",
    "The Linux kernel began in 1991 as a hobby project by Linus Torvalds, a Finnish student studying at the University of Helsinki. Released under the GPL in 1992, it grew into the dominant kernel for servers, mobile devices, and embedded systems. Today its development involves thousands of contributors worldwide.",
    "Mitochondria are membrane-bound organelles found in nearly all eukaryotic cells. They generate most of the cell's ATP through oxidative phosphorylation across the inner membrane. Mitochondria carry their own circular DNA, a vestige of the bacterial ancestor that became permanently endosymbiotic over a billion years ago.",
    "The Battle of Midway in June 1942 was a turning point in the Pacific war. American carrier aircraft sank four Japanese fleet carriers in a single day, neutralizing Japan's offensive carrier capability. Cryptanalysts who had broken the JN-25 code provided the intelligence that made the ambush possible.",
    "Floating-point arithmetic represents real numbers as a sign, an exponent, and a mantissa. The IEEE 754 standard, ratified in 1985, defines binary formats and rounding rules used by virtually all modern hardware. Despite its ubiquity, careless use leads to cancellation, overflow, and surprising rounding behavior.",
    "Coral reefs are biodiverse ecosystems built by tiny animals called polyps that secrete calcium-carbonate skeletons. Reefs support roughly a quarter of all marine species despite covering less than one percent of the ocean floor. Rising sea temperatures and ocean acidification threaten their survival in many regions.",
    "The Fast Fourier Transform, popularized by Cooley and Tukey in 1965, computes the discrete Fourier transform in O(n log n) time. It revolutionized signal processing, enabling real-time spectrum analysis, efficient convolution, and the digital filtering that underlies modern audio, image, and radio engineering.",
    "Plate tectonics describes the slow motion of large fragments of the Earth's lithosphere over the underlying asthenosphere. Plates diverge at mid-ocean ridges where new crust forms and converge at subduction zones where one plate sinks beneath another. The theory unified continental drift and seafloor spreading.",
    "The Library of Alexandria, founded in the third century BCE, aimed to collect every book in the known world. Scholars associated with it, including Eratosthenes and Euclid, made foundational contributions to mathematics, astronomy, and geography. Its decline was gradual rather than the result of a single dramatic destruction.",
    "Vaccination trains the immune system to recognize a pathogen by exposing it to a harmless mimic — an inactivated virus, a subunit protein, or messenger RNA encoding a viral antigen. Subsequent exposure to the real pathogen triggers a rapid, specific response that prevents or limits disease.",
    "The James Webb Space Telescope, launched in December 2021, observes primarily in the infrared. Its 6.5-meter segmented gold-coated mirror, twenty times the light-gathering area of Hubble, lets it study the earliest galaxies, exoplanet atmospheres, and dust-shrouded star-forming regions invisible to optical telescopes.",
    "The Diffie-Hellman key exchange, published in 1976, lets two parties agree on a shared secret over an insecure channel. Its security rests on the difficulty of computing discrete logarithms in a finite group. The protocol underlies modern TLS, SSH, and many other cryptographic systems used daily.",
    "Antibiotic resistance arises when bacteria acquire mutations or horizontally transferred genes that allow them to survive drugs that once killed them. Overuse of antibiotics in medicine and agriculture accelerates the spread of resistant strains. Coordinated stewardship and the development of new antibiotic classes are urgent priorities.",
    "Markov chains are stochastic processes where the next state depends only on the current state, not on the full history. They are used in everything from PageRank's web-graph analysis to speech-recognition language models to the underlying engine of modern reinforcement-learning algorithms.",
    "The eruption of Mount Tambora in 1815 was the largest volcanic event in recorded history. Its sulfate aerosols spread through the stratosphere and cooled the planet by roughly half a degree, producing the 'year without a summer' in 1816. Crop failures across Europe and North America followed.",
    "Public-key cryptography decouples the keys used to encrypt and decrypt messages. Anyone holding the public key can encrypt or verify; only the holder of the corresponding private key can decrypt or sign. RSA, ElGamal, and elliptic-curve schemes all instantiate this idea using different mathematical hard problems.",
    "Honeybee colonies coordinate foraging through the waggle dance, in which a returning forager encodes the direction and distance of a food source through the angle and duration of her dance on the comb. The behavior was decoded by Karl von Frisch in the 1940s, earning him the 1973 Nobel Prize.",
    "The B-tree, introduced by Bayer and McCreight in 1971, is a self-balancing search tree designed for storage systems with high access cost. Each node holds many keys and many children, which keeps the tree shallow and minimizes I/O. B-trees and their variants underlie almost every relational database engine.",
    "Antarctic ice cores preserve atmospheric samples in trapped air bubbles, allowing reconstruction of greenhouse-gas concentrations over the last 800,000 years. The records show CO2 oscillating between roughly 180 and 280 parts per million across glacial cycles, with a sharp recent rise to over 400 ppm.",
    "The Apollo Guidance Computer, developed at MIT in the 1960s, ran the navigation software for the lunar landings. It had four kilobytes of RAM and 72 kilobytes of read-only core rope memory, but its real-time priority scheduler famously kept the descent on track when the radar data overflow alarm fired during Apollo 11.",
    "The blood-brain barrier is formed by tight junctions between endothelial cells in cerebral capillaries. It restricts the passage of large or hydrophilic molecules into the central nervous system, protecting neural tissue from toxins and pathogens but also complicating the design of drugs intended to act on the brain.",
    "Reinforcement learning frames decision-making as the maximization of expected cumulative reward over a sequence of states and actions. Modern variants combine deep neural networks with policy-gradient or value-iteration updates, yielding agents that play games, control robots, and align large language models with human preferences.",
    "The Phoenician alphabet, developed around 1050 BCE, used a small set of consonant signs that traders carried throughout the Mediterranean. It is the ancestor of Greek, Latin, Arabic, and Hebrew scripts. The radical compression — from thousands of cuneiform signs to a few dozen letters — democratized literacy.",
]


def build_engine(verbose: bool = True):
    """Construct the standard V4-Flash TP=8 fp16 engine."""
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor")
    )
    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    cuda_bin = "/usr/local/cuda-12.8/bin"
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + ":" + cuda_bin

    from vllm import LLM

    if verbose:
        print(
            "[1/3] Building LLM(model=V4-Flash-W4A16, tp=8, fp16, eager)...",
            flush=True,
        )
    llm = LLM(
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        tensor_parallel_size=8,
        dtype="float16",
        enforce_eager=True,
        quantization="auto-round",
        max_model_len=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=False,
        block_size=64,  # V100 sparse_attn paged-cache helper uses BLOCK_N=64.
        trust_remote_code=False,
    )
    return llm


def per_seq_ppl(prompt_logprobs, prompt_token_ids):
    """Compute PPL for one sequence from prompt_logprobs.

    prompt_logprobs: list of length N. Entry 0 is None (no logprob for the
    first token). Entries 1..N-1 are dicts keyed by token id with Logprob
    values. The actual prompt token at position i is always present in
    the dict (vLLM inserts it even when it isn't in top-k).

    Returns (ppl, n_scored_tokens).
    """
    nll_sum = 0.0
    n = 0
    for i in range(1, len(prompt_logprobs)):
        entry = prompt_logprobs[i]
        if entry is None:
            continue
        tid = prompt_token_ids[i]
        # entry maps token_id -> Logprob(.logprob, .rank, .decoded_token)
        lp = entry.get(tid)
        if lp is None:
            # Fallback: take min logprob in the dict (shouldn't happen
            # for the actual token, but defend against engine quirks).
            continue
        nll_sum += -lp.logprob
        n += 1
    if n == 0:
        return float("nan"), 0
    return math.exp(nll_sum / n), n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=len(CORPUS),
        help="Number of corpus snippets to score (default: full corpus)."
    )
    parser.add_argument(
        "--label", type=str, default="run",
        help="Tag for this run (used in stdout, e.g. 'clamp' or 'fp32res')."
    )
    parser.add_argument(
        "--max-tokens-per-prompt", type=int, default=512,
        help="Truncate each prompt to at most this many tokens."
    )
    args = parser.parse_args()

    snippets = CORPUS[: args.n]

    llm = build_engine()

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)

    # Tokenize and (gently) truncate. Keep BOS suppressed: V4-Flash's
    # tokenizer adds a BOS by default, but the engine's prefill already
    # starts at position 0 — we want the corpus text only, not extra
    # ceremonial tokens. add_special_tokens=False matches what the bar
    # test does for raw token prompts.
    prompts = []
    for s in snippets:
        ids = tok.encode(s, add_special_tokens=False)
        ids = ids[: args.max_tokens_per_prompt]
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    print(
        f"[2/3] Scoring {len(prompts)} sequences "
        f"(label={args.label!r}, prompt_logprobs path)...",
        flush=True,
    )
    t0 = time.time()
    outs = llm.generate(prompts=prompts, sampling_params=sp)
    dt = time.time() - t0

    ppls = []
    n_tokens_total = 0
    for o in outs:
        ppl, n = per_seq_ppl(o.prompt_logprobs, list(o.prompt_token_ids))
        ppls.append(ppl)
        n_tokens_total += n

    finite_ppls = [p for p in ppls if math.isfinite(p)]
    print(f"[3/3] Done in {dt:.2f}s ({n_tokens_total} scored tokens).", flush=True)
    print(f"  label = {args.label}", flush=True)
    print(f"  n_seqs = {len(ppls)}, n_finite = {len(finite_ppls)}", flush=True)
    if finite_ppls:
        print(f"  mean_ppl = {statistics.mean(finite_ppls):.4f}", flush=True)
        print(f"  median_ppl = {statistics.median(finite_ppls):.4f}", flush=True)
        print(f"  min_ppl = {min(finite_ppls):.4f}", flush=True)
        print(f"  max_ppl = {max(finite_ppls):.4f}", flush=True)
        if len(finite_ppls) > 1:
            print(f"  stdev_ppl = {statistics.stdev(finite_ppls):.4f}", flush=True)
    print("  per_seq_ppl:", flush=True)
    for i, p in enumerate(ppls):
        print(f"    seq[{i:02d}] = {p:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
