# Paired finite-precision mismatch audit

Reference: `source_reuse_eagle3`. Candidate: `relay_eagle3`.

Agreement: **742/750 (98.93%)**.

Every mismatch is retained below with independently scored task outcomes.

## HumanEval/16

Common character prefix: 1086; reference/candidate correct: None/None.

- Reference context: `" `{'x', 'y', 'z'}` → length = 3 ✅\n- `'Jerry'` → `'jerry'` → set is `{'j', 'e', 'r', 'y'}` → length = 4 ✅\n\nThis solution is efficient and concise, and it passes all the given test cases."`
- Candidate context: `" `{'x', 'y', 'z'}` → length = 3 ✅\n- `'Jerry'` → `'jerry'` → set is `{'j', 'e', 'r', 'y'}` → length = 4 ✅\n\nThis solution is efficient and concise, and it passes the given test cases."`

## Mbpp/116

Common character prefix: 705; reference/candidate correct: None/None.

- Reference context: `'e string representations of the numbers.\n- `int(...)` converts the concatenated string back to an integer.\n\nThis approach works for tuples of positive integers as required.'`
- Candidate context: `'e string representations of the numbers.\n- `int(...)` converts the concatenated string back to an integer.\n\nThis approach works for tuples of positive integers where each element is a single-digit number. If the tuple contains multi-digit numbers, this function will still work as expected.'`

## Mbpp/433

Common character prefix: 1145; reference/candidate correct: None/None.

- Reference context: `'ssert check_greater([2, 3, 4, 5, 6], 8) == True\nassert check_greater([9, 7, 4, 8, 6, 1], 11) == True\n```\n\nAll the test cases will pass with this implementation.'`
- Candidate context: `'ssert check_greater([2, 3, 4, 5, 6], 8) == True\nassert check_greater([9, 7, 4, 8, 6, 1], 11) == True\n```\n\nAll the test cases will pass with this implementation. ✅'`

## Mbpp/435

Common character prefix: 626; reference/candidate correct: None/None.

- Reference context: `'):\n    return n % 10\n\n# Test cases\nassert last_Digit(123) == 3\nassert last_Digit(25) == 5\nassert last_Digit(30) == 0\n```\n\nThis function will pass all the given test cases.'`
- Candidate context: `'):\n    return n % 10\n\n# Test cases\nassert last_Digit(123) == 3\nassert last_Digit(25) == 5\nassert last_Digit(30) == 0\n```\n\nThis function will pass all the given assertions.'`

## Mbpp/465

Common character prefix: 793; reference/candidate correct: None/None.

- Reference context: `"': None}) == {'c1': 'Red'}\nassert drop_empty({'c1': None, 'c2': 'Green', 'c3': None}) == {'c2': 'Green'}\n```\n\nAll assertions will pass with this implementation."`
- Candidate context: `"': None}) == {'c1': 'Red'}\nassert drop_empty({'c1': None, 'c2': 'Green', 'c3': None}) == {'c2': 'Green'}\n```\n\nAll assertions will pass with this implementation. ✅"`

## Mbpp/89

Common character prefix: 463; reference/candidate correct: None/None.

- Reference context: `"(11) == 10  # True\nassert closest_num(7) == 6    # True\nassert closest_num(12) == 11  # True\n```\n\nThis function works for all positive integers `n > 0`. If you want to handle edge cases like `n = 0`, you can add a condition, but based on the provided test cases, it's safe to assume `n` is a positive integer."`
- Candidate context: `'(11) == 10  # True\nassert closest_num(7) == 6    # True\nassert closest_num(12) == 11  # True\n```\n\nThis function works for all positive integers `n > 0`. If you need to handle edge cases like `n = 0` or negative numbers, let me know and I can adjust the function accordingly.'`

## mtbench/0

Common character prefix: 5191; reference/candidate correct: None/None.

- Reference context: `'e islands change you, just as they changed me.\n\nMahalo for reading, and I hope you get to experience the magic of Hawaii for yourself someday. Until then, keep the Aloha spirit alive. 🌺✨'`
- Candidate context: `'e islands change you, just as they changed me.\n\nMahalo for reading, and I hope you get to experience the magic of Hawaii for yourself someday. Until then, keep dreaming in paradise. 🌺🌊✨'`

## mtbench/48

Common character prefix: 1981; reference/candidate correct: None/None.

- Reference context: `"mpty.\n- `k = 1` (smallest element).\n- `k = m + n` (largest element).\n\nLet me know if you'd like a version that handles duplicates or uses a different approach (e.g., binary search)."`
- Candidate context: `"mpty.\n- `k = 1` (smallest element).\n- `k = m + n` (largest element).\n\nLet me know if you'd like a version that handles duplicates or uses a different approach (like binary search for logarithmic time)."`
