# Paired finite-precision mismatch audit

Reference: `source_reuse_eagle3`. Candidate: `relay_eagle3`.

Agreement: **496/500 (99.20%)**.

Every mismatch is retained below with independently scored task outcomes.

## math500/193

Common character prefix: 1026; reference/candidate correct: True/True.

- Reference context: `"\\frac{1}{2}\n$$\n\nThat's **4 distinct rational roots**.\n\n---\n\n### Step 3: Final Answer\n\n$$\n\\boxed{4}\n$$\n\nThere are **4 different possible rational roots** of the given polynomial."`
- Candidate context: `"\\frac{1}{2}\n$$\n\nThat's **4 distinct rational roots**.\n\n---\n\n### Step 3: Final Answer\n\n$$\n\\boxed{4}\n$$\n\nThere are **4 different possible rational roots** of the polynomial."`

## math500/284

Common character prefix: 2118; reference/candidate correct: False/False.

- Reference context: `'{2}\n$$\n\n$$\nx = \\frac{6 \\pm 4\\sqrt{2}}{2} = 3 \\pm 2\\sqrt{2}\n$$\n\n---\n\n### Step 5: Final Answer\n\nSo the solutions are:\n\n$$\n\\boxed{3 + 2\\sqrt{2},\\ 3 - 2\\sqrt{2}}\n$$'`
- Candidate context: `'{2}\n$$\n\n$$\nx = \\frac{6 \\pm 4\\sqrt{2}}{2} = 3 \\pm 2\\sqrt{2}\n$$\n\n---\n\n### Step 5: Final Answer\n\nSo the solutions are:\n\n$$\n\\boxed{3 + 2\\sqrt{2},\\ 3 - 2\\sqrt{2}}\n$$\n\nThese are the values of $ x $ that satisfy the original equation.'`

## math500/5

Common character prefix: 1224; reference/candidate correct: True/True.

- Reference context: `'ides**, each of length $ s = 7 $ inches.\n\nSo, the perimeter of the hexagon is:\n\n$$\n6 \\times 7 = 42 \\text{ inches}\n$$\n\n---\n\n### ✅ Final Answer:\n\n$$\n\\boxed{42}\n$$'`
- Candidate context: `'ides**, each of length $ s = 7 $ inches.\n\nSo, the perimeter of the hexagon is:\n\n$$\n6 \\times 7 = 42 \\text{ inches}\n$$\n\n---\n\n### ✅ Final Answer:\n\n$$\n\\boxed{42}\n$$ inches.'`

## math500/8

Common character prefix: 822; reference/candidate correct: True/True.

- Reference context: `'ring:\n\n$$\n117 = 9 \\times 13\n$$\n\nSo:\n\n$$\n\\sqrt{117} = \\sqrt{9 \\times 13} = \\sqrt{9} \\cdot \\sqrt{13} = 3\\sqrt{13}\n$$\n\n### ✅ Final Answer:\n$$\n\\boxed{3\\sqrt{13}}\n$$'`
- Candidate context: `'ring:\n\n$$\n117 = 9 \\times 13\n$$\n\nSo:\n\n$$\n\\sqrt{117} = \\sqrt{9 \\times 13} = \\sqrt{9} \\cdot \\sqrt{13} = 3\\sqrt{13}\n$$\n\n### ✅ Final Answer:\n$$\n\\boxed{3\\sqrt{13}}\n$$ units.'`
