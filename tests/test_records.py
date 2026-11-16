import unittest
from verify_results import validate_timings


class RecordTests(unittest.TestCase):
    def test_incomplete_and_nonfinite_observations_rejected(self):
        valid = {name: [1.0] * 30 for name in ("prepacked", "online-a", "online-both")}
        validate_timings(valid, 30)
        for invalid in (float("nan"), float("inf"), -1, 0, True):
            corrupted = {name: values.copy() for name, values in valid.items()}
            corrupted["online-a"][0] = invalid
            with self.assertRaises(ValueError):
                validate_timings(corrupted, 30)
        valid["online-both"].pop()
        with self.assertRaises(ValueError):
            validate_timings(valid, 30)


if __name__ == "__main__":
    unittest.main()
