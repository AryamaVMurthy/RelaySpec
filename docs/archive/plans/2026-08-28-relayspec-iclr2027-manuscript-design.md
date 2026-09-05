# RelaySpec ICLR 2027 Manuscript Design

## Purpose

Create an anonymous ICLR 2027 submission that explains one clear result:
an existing feature-conditioned speculative proposer can be reused with a new
compatible target model by learning only the tensor interface that the
proposer consumes. This removes the original source-model trunk from the
repeated inference loop.

The manuscript must use the official ICLR 2027 style. The main text must fit
within nine pages. References and appendices follow the official rules.

## Paper question

The introduction asks one concrete question:

> Can we reuse a frozen speculative proposer with a different target model
> without retraining the proposer or running its original source model during
> every generation cycle?

The answer is RelaySpec. The target model already computes hidden states while
verifying proposed tokens. A small linear map converts five of those states
into the tensor expected by the old proposer. The proposer stays frozen. The
target still checks every proposed token.

## Claim hierarchy

The paper makes four claims in this order.

1. Reusing an old feature-conditioned proposer can require a second model in
   the repeated inference path. In the measured baselines, this source trunk
   takes 27 to 40 percent of request time.
2. A target-specific linear interface map can replace that source trunk for
   both DFlash and EAGLE-3 proposers with Qwen3-8B and Qwen3-14B targets.
3. Speed depends on two measured quantities: the time removed and the
   accepted-token rate after translation. A simple latency equation predicts
   all 16 observed task directions with 0.34 percent mean relative error.
4. The method improves the four main MATH-500 cells by 1.082 to 1.483 times,
   keeps paired official accuracy unchanged, saves 20 to 31 percent peak
   memory in the isolated EAGLE measurements, and fits in under two minutes.

The paper does not claim universal raw speedup, production serving throughput,
cross-architecture compatibility, or byte-identical BF16 output for EAGLE-3.

## Main-paper structure

The nine-page main text uses the following structure.

1. **Introduction:** the deployment problem, the proposed replacement, the
   measured result, and four contributions.
2. **Background and problem:** how feature-conditioned speculative decoding
   works, what source reuse executes, and the exact scope of compatibility.
3. **Method:** the source path, the relay path, the training loss, and the
   complete inference procedure.
4. **When the relay helps:** the acceptance-aware latency equation, its
   break-even condition, and the conservative workload-level provider rule.
5. **Experimental setup:** models, frozen components, data splits, tasks,
   quality metrics, timing protocol, hardware, and confidence intervals.
6. **Results:** main MATH table, cross-task result, predicted-versus-observed
   plot, memory and adaptation cost, and only the two design ablations needed
   to support the final method.
7. **Related work:** organize by the exact difference from RelaySpec rather
   than listing papers.
8. **Limitations and conclusion:** state the measured scope and summarize the
   result without adding new claims.
9. **Required AI-use statement:** use the official ICLR 2027 requirement and
   describe actual assistance accurately.

Detailed tables, all 16 matrix rows, complete mismatch audits, per-position
acceptance, data provenance, implementation details, and additional ablations
go in the appendix.

## Figures and tables

The main paper contains four visual elements.

1. **System diagram:** one panel shows source reuse and one shows RelaySpec.
   It makes the removed Qwen3-4B trunk visually explicit.
2. **Main result table:** two proposer families by two target scales. It shows
   source and relay throughput, paired speed interval, official accuracy, and
   exact text agreement.
3. **Predicted versus observed plot:** all 16 task cells, with the diagonal and
   labels for negative cells. This supports the latency model directly.
4. **Resource table:** relay parameters, fit time, target-specific proposer
   comparison, and isolated memory saving.

A small appendix plot shows accepted-prefix survival by draft position. A
second appendix table gives the complete cross-task matrix.

All plots are generated from machine-readable result artifacts. No value is
typed into a figure by hand.

## Writing rules

- Use short, direct sentences.
- Explain the procedure before giving it a name.
- Define every symbol immediately.
- Use ordinary words when a technical label adds no precision.
- Do not use em dashes or semicolons.
- Do not use claims such as "general," "lossless," or "exact" without stating
  the exact scope.
- Do not repeat the same result in the abstract, introduction, results, and
  conclusion with different wording.
- Every paragraph must answer what is done, why it is needed, or how it is
  measured.

## Verification design

The final paper passes six independent checks.

1. **Format:** official ICLR 2027 style, anonymous authors, at most nine pages
   before references, and required policy sections.
2. **Numbers:** every result in the manuscript matches the JSON artifacts.
3. **Citations:** every citation key exists and every venue status is correct.
4. **Writing:** no em dash, semicolon, banned vague phrase, unexplained symbol,
   repeated paragraph, placeholder, or unsupported superlative.
5. **Overlap:** no duplicated paragraph within the manuscript and zero fit to
   evaluation prompt overlap in the experimental data.
6. **Rendering:** all pages are rendered to images and checked for clipping,
   overlap, small labels, broken references, and page-limit compliance.

## Deliverables

- `paper/iclr2027/relayspec_iclr2027.tex`
- official ICLR 2027 style and bibliography files
- generated figures under `paper/iclr2027/figures/`
- generated numerical macros and appendix tables
- `output/pdf/RelaySpec_ICLR_2027.pdf`
- a manuscript QA report with format, claim, citation, style, and rendering
  checks
