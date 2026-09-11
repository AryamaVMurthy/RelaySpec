import json
from concurrent.futures import ThreadPoolExecutor
from experiments.auf_vllm.pilot_data import write


def test_parallel_json_writes_are_complete_and_do_not_share_temporary_paths(tmp_path):
    path = tmp_path / 'shared.json'
    def worker(index):
        write(path, {'index': index, 'payload': [index] * 1000})
        observed = json.loads(path.read_text())
        assert observed['payload'] == [observed['index']] * 1000
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(64)))
    assert set(tmp_path.iterdir()) == {path}
