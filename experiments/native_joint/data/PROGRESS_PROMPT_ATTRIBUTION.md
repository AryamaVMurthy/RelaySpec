# Calibration prompt attribution

Adapted from Code Alpaca by Sahil Chaudhary (2023).

Source: https://github.com/sahil280114/codealpaca

Pinned revision: 2f78ddc5c682ed6738ad092bbbfa59ba915afcb0

Data license: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/

Changes: Python-keyword instruction subset, deterministic ordering, lexical overlap filtering, joined instruction/input, split and metadata. Synthetic output correctness is not established.


General-instruction records are adapted from Databricks Dolly 15k.

Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k

License: CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/

Pinned revision: bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a

Changes: deterministic subset selection and lexical overlap filtering; instruction and context joined, metadata added, and selected records combined with the separately sourced Numina math pool.


Further changes for this standalone study: prompt-only subsets, additional lexical filtering, and independent native-model rollouts. Original solutions are not used.
