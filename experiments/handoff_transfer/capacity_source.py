"""Explicit rank-only transformation for the separate capacity ablation."""
def rank_variant(source, rank):
    assert rank in (8, 16, 32, 56, 128, 256)
    replacements = [
        ("VARIANTS=['fusion_r56','five_maps']", "VARIANTS=['fusion_capacity']"),
        ('LoraConfig(r=56,lora_alpha=56,', f'LoraConfig(r={rank},lora_alpha={rank},'),
        ('assert n==(56*', f'assert n==({rank}*'),
        ("'rank':56 if", f"'rank':{rank} if"),
        ("'alpha':56 if", f"'alpha':{rank} if"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, f'Upstream rank contract changed: {old}'
        source = source.replace(old, new)
    return source.replace("'fusion_r56'", "'fusion_capacity'")
