# Native baseline availability checked September12

The current exact32e1b8 suite has AR/original/ZIP references, not target-native
DFlash references. Its AUF speedups cannot be described as gains over native.

| Target | Official native DFlash checkpoint | Local asset |
|---|---|---|
| Qwen3-8B | z-lab/Qwen3-8B-DFlash-b16 | transfer-reproduction-20260907/work/models/8b/draft |
| Llama3.1-8B-Instruct | z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat | relayspec-auf-20260911/models/llama8-draft |
| Llama3.2-3B-Instruct | No official released checkpoint verified | None in study models |

Primary sources:
- https://github.com/z-lab/dflash/blob/main/README.md (supported-model listing)
- https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16
- https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat

The Llama8B native asset is the same source drafter used in Llama8B→3B
retargeting. It is a native control for the target of Qwen4B→Llama8B. It is
not native to Llama3.2-3B. Lack of a verified downloadable3B checkpoint here
is not proof that nobody has trained one. A new native draft should not be
silently trained as an extra baseline outside the agreed experiment scope.

A relevant primary paper surfaced in the3B availability search:
https://arxiv.org/html/2606.11552v2
Teaching Diffusion to Speculate Left-to-Right (Whalen, Ito, Sakamoto). Its
abstract studies positional weighting, a first-error focal loss and a joint
prefix chain loss for diffusion drafters. This requires a precise comparison
in the later AUF related-work/theory audit; the search does not establish
identity with our AUF or verify released checkpoint availability. No new
experiment is authorized or launched by this note.
