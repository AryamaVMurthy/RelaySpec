from collections import Counter
from experiments.auf_vllm.fit_updates import record_indices


def test_sampler_fixed_work_and_resume():
    for count in [16,32,64,512,4096]:
        steps=max(2,count//32)
        sequence=[record_indices(count,step,42) for step in range(steps)]
        assert all(len(x)==32 for x in sequence)
        # Resuming requires only the global step, not a reset RNG/epoch stream.
        assert sequence[1:]==[record_indices(count,step,42) for step in range(1,steps)]
        counts=Counter(index for batch in sequence for index in batch)
        assert len(counts)==count
        assert max(counts.values())==min(counts.values())
