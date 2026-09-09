import json
from pathlib import Path
import tempfile
import unittest
from analyze import summarize


class AuditTests(unittest.TestCase):
    def fixture(self, path):
        methods = ["ar", "native", "source", "relay"]
        rows = []
        t = 1.
        for repeat in range(2):
            for i,b in enumerate(["gsm8k","math500","humaneval","mtbench"]):
                for m in methods:
                    seconds = {"ar":4., "native":1., "source":2., "relay":1.}[m]
                    rows.append(dict(method=m,problem_id=str(i),repeat=repeat,benchmark=b,seconds=seconds,
                        monotonic_start=t,monotonic_stop=t+seconds,tokens=[1]*8,output_tokens=8,
                        acceptance_lengths=[1,2],capped=False,peak_allocated=2**30,peak_reserved=2**31,incremental_peak_allocated=1024))
                    t += seconds
        samples = [dict(monotonic=i,devices=[{"power.draw":"100","utilization.gpu":"80","memory.used":"[N/A]"}],
                        process_memory={"VmRSS_bytes":2**30},system_memory={"MemAvailable_bytes":2**34}) for i in range(int(t)+2)]
        for name, value in [("complete.json",dict(status="pass",profiling=False,repeats=2,rows=len(rows))),
                            ("provenance.json",dict(methods=methods,cap=128))]:
            (path/name).write_text(json.dumps(value))
        (path/"evaluation.jsonl").write_text("\n".join(map(json.dumps,rows)))
        (path/"telemetry.jsonl").write_text("\n".join(map(json.dumps,samples)))
        return rows

    def test_ratios_energy_missing_counters_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); self.fixture(path)
            r=summarize(path)["overall"]["relay"]
            self.assertEqual(r["tps"],8.)
            self.assertEqual(r["ratios"]["source"],{"ratio":2.,"ci95":[2.,2.]})
            self.assertEqual(r["device_joules_per_token"],12.5)
            self.assertEqual(r["exact_ar"],4)
            self.assertIsNone(r["gpu_metrics"]["memory.used"]["mean"])

    def test_missing_row_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); rows=self.fixture(path)
            (path/"evaluation.jsonl").write_text("\n".join(map(json.dumps,rows[:-1])))
            with self.assertRaises(AssertionError): summarize(path)

    def test_changed_repeat_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); rows=self.fixture(path); rows[-1]["tokens"]=[2]*8
            (path/"evaluation.jsonl").write_text("\n".join(map(json.dumps,rows)))
            with self.assertRaises(AssertionError): summarize(path)


if __name__ == "__main__": unittest.main()
