"""Assign development strata using input-only benchmark attributes."""


def input_strata(record, input_tokens, protocol):
    if record.get("benchmark") != "math500":
        raise ValueError("MATH task labels required")
    metadata = record["metadata"]
    level = metadata["level"]
    subject = metadata["subject"]
    if type(level) is not int or level not in range(1, 6) or not subject:
        raise ValueError("invalid original difficulty or subject label")
    if type(input_tokens) is not int or input_tokens < 1:
        raise ValueError("positive shared input token count required")
    difficulty = [k for k, levels in protocol["difficulty"].items() if level in levels]
    lengths = [
        k
        for k, (low, high) in protocol["input_token_bins"].items()
        if low <= input_tokens and (high is None or input_tokens <= high)
    ]
    if len(difficulty) != 1 or len(lengths) != 1:
        raise ValueError("input strata must be disjoint and exhaustive")
    return {"difficulty": difficulty[0], "subject": subject, "input_length": lengths[0]}
