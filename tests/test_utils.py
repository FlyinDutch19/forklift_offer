# test_utils.py
import unittest
from utils import safe_float, parse_battery_size, size_within_limit

class TestUtils(unittest.TestCase):
    def test_safe_float(self):
        self.assertEqual(safe_float(1.2), 1.2)
        self.assertEqual(safe_float("2.5"), 2.5)
        self.assertEqual(safe_float("N/A"), 0.0)
        self.assertEqual(safe_float("-"), 0.0)
        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float(""), 0.0)
    def test_parse_battery_size(self):
        self.assertEqual(parse_battery_size("900x400x600"), (900.0, 400.0, 600.0))
        self.assertEqual(parse_battery_size("900*400*600"), (900.0, 400.0, 600.0))
        self.assertEqual(parse_battery_size("900×400×600"), (900.0, 400.0, 600.0))
        self.assertEqual(parse_battery_size("900 X 400 X 600"), (900.0, 400.0, 600.0))
        self.assertIsNone(parse_battery_size("900x400"))
        self.assertIsNone(parse_battery_size(""))
    def test_size_within_limit(self):
        self.assertTrue(size_within_limit("900x400x600", (900,400,600)))
        self.assertTrue(size_within_limit("800x300x500", (900,400,600)))
        self.assertFalse(size_within_limit("1000x500x700", (900,400,600)))
        self.assertTrue(size_within_limit("900x400x600", None))

if __name__ == "__main__":
    unittest.main()
