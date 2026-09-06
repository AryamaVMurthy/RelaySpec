# RelaySpec positioning revision — 6 September 2026

The revised paper explains why a small interface is useful: it reuses an inherited drafter's learned behavior with a new target, removes source-transformer execution, and needs little fitting data in the tested setting.

- Expanded related work from approximately 186 to 556 whitespace-delimited words, covering inherited drafters, target-independent drafting, dynamic steering, distillation/adaptation, frozen interfaces and execution methods.
- Rewrote the abstract, introduction, contributions and conclusion to connect the mechanism to practical value and distinguish the contribution from prior linear connectors and cross-target reuse.
- Made the completed 8B capacity finding explicit: dense mapping has the highest mapper throughput across each longer-output comparison. MLPs, parameter-matched factorizations, longer fitting and regularization provide evidence for this choice, with their limits retained.
- Added main Table 4 with all three warm fitting budgets and four adaptation methods. Feature regression wins every tested budget against connector CE and both LoRA seeds. The table regenerates from the audited evidence.
- Made RelaySpec, PARD and frozen SD-square throughput visible together in main text, retaining the different-runtime scope.
- Preserved held-out confirmation, quality uncertainty, composition tradeoffs and unresolved 14B optimization behavior. No experiments or numerical results were changed for this revision.

The compiled paper has nine main-text pages and 37 pages including references and appendix, with 42 resolved citations. All pages were visually inspected in color and all distinct grayscale renderings were reviewed. The final automated check results are recorded in ICLR_MANUSCRIPT_QA.md.
