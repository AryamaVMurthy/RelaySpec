"""Check telemetry integration against independently known energy values."""
import unittest
from analyze import energy, numeric


def sample(t, power):
    return {"monotonic": t, "devices": [{"power.draw": power}]}


class MetricsTests(unittest.TestCase):
    def test_constant_power(self):
        self.assertAlmostEqual(energy([sample(0, 50), sample(1, 50), sample(2, 50)], .25, 1.75), 75)

    def test_ramp_interpolated_endpoints(self):
        self.assertAlmostEqual(energy([sample(0, 10), sample(1, 20), sample(2, 30)], .5, 1.5), 20)

    def test_unsupported_power_is_not_zero(self):
        self.assertIsNone(numeric("[N/A]"))
        self.assertIsNone(energy([sample(0, 50), sample(1, "N/A"), sample(2, 50)], .25, 1.75))

    def test_failed_sample_is_not_discarded(self):
        self.assertIsNone(energy([sample(0, 50), {"monotonic": 1, "devices": []}, sample(2, 50)], .25, 1.75))

    def test_missing_temporal_coverage(self):
        self.assertIsNone(energy([sample(0, 50), sample(1, 50)], 0, 2))
        self.assertIsNone(energy([sample(0, 50), sample(5, 50)], 1, 4))


if __name__ == "__main__":
    unittest.main()
