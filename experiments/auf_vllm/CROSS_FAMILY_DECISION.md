# Exact AUF cross-family compatibility decision

The original Qwen4-to-Llama8 greedy text bridge does not supply the target-token
probabilities required by the supplied AUF loss. This was checked dynamically
with the pinned tokenizers in CPU job31341, not inferred from model names.

- 41,739 source tokens retokenize into multiple target tokens;14 become empty
  when the existing bridge drops special tokens.
- The singleton route reaches108,589 of128,256 target vocabulary entries;
 19,667 target entries, including target EOS128009, have no singleton support.
- 1,410 archived lookup entries differ from decoding and re-encoding that
  individual source token. This count includes lossy partial-byte decoding.
- Even fully lookup-covered ordinary text changes target boundaries: “The
  answer is42.” has7 source positions and6 target positions. The token lookup
  therefore differs from the text-reencoding route.

A normalized distribution over only supported singleton tokens would assign
zero probability to unsupported valid labels. Dropping those labels changes the
specified validity mask; smoothing or adding a learned target-vocabulary head
changes the method. Variable-length sequence probability mapping would also
need an explicitly different training/proposer contract. Greedy text proposals
can still be verified by the target, but that does not provide the supplied
per-target-position differentiable AUF objective.

Apply the user-authorized fallback in the study plan: defer heterogeneous-
vocabulary AUF as a separate architecture extension. The new study continues
with Qwen4-to-Qwen8, Qwen4-to-Qwen14, and Llama8-to-Llama3. Keep old cross-family
results under their original method/runtime; do not present them as new AUF
measurements. This is a limitation of the audited bridge, not an impossibility
claim about all cross-family speculative decoding.

The core new campaign therefore contains21 main/robustness fits and120 trained-
method128-request evaluation units, excluding baselines, sweeps, profiling and
backend checks. The exact Qwen scope statement remains:

Our reported Qwen3 experiments use a shared tokenizer and token-ID vocabulary;
the mapper adapts hidden representations or the fusion interface, not vocabulary
IDs. Consequently, these results do not demonstrate heterogeneous-vocabulary
DFlash support.
