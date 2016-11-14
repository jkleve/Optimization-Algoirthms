import unittest

import sys
sys.path.append("../utils")

from oa_utils import gen_random_numbers

class test_random_numbers(unittest.TestCase):
    def test_random(self):
        bounds = [(-10,10), (-10,10)]
        arr = gen_random_numbers(bounds)

        self.assertIsNotNone(arr)
        self.assertIsInstance(arr, list)
        self.assertIsInstance(arr[0], float)
        self.assertIsInstance(arr[1], float)

if __name__ == "__main__":
    unittest.main()
