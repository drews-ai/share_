#!/usr/bin/env python3
"""Regression tests for Sparkov calibration integration."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sharewith_thesis_algo as algo  # noqa: E402


HEADERS = [
    "",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


class SparkovIntegrationTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(rows)

    def test_profile_derivation_and_weights_normalize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_dir = Path(td)
            rows_train = [
                {
                    "": "0",
                    "trans_date_trans_time": "2019-01-01 00:00:18",
                    "cc_num": "1111",
                    "merchant": "fraud_A",
                    "category": "grocery_pos",
                    "amt": "12.0",
                    "first": "A",
                    "last": "A",
                    "gender": "F",
                    "street": "x",
                    "city": "x",
                    "state": "NC",
                    "zip": "11111",
                    "lat": "35.0",
                    "long": "-80.0",
                    "city_pop": "1000",
                    "job": "dev",
                    "dob": "1990-01-01",
                    "trans_num": "t1",
                    "unix_time": "1546300800",
                    "merch_lat": "35.1",
                    "merch_long": "-80.1",
                    "is_fraud": "0",
                },
                {
                    "": "1",
                    "trans_date_trans_time": "2019-01-01 01:00:18",
                    "cc_num": "1111",
                    "merchant": "fraud_B",
                    "category": "home",
                    "amt": "50.0",
                    "first": "A",
                    "last": "A",
                    "gender": "F",
                    "street": "x",
                    "city": "x",
                    "state": "NC",
                    "zip": "11111",
                    "lat": "35.0",
                    "long": "-80.0",
                    "city_pop": "1000",
                    "job": "dev",
                    "dob": "1990-01-01",
                    "trans_num": "t2",
                    "unix_time": "1546304400",
                    "merch_lat": "35.2",
                    "merch_long": "-80.2",
                    "is_fraud": "0",
                },
                {
                    "": "2",
                    "trans_date_trans_time": "2019-01-02 01:00:18",
                    "cc_num": "2222",
                    "merchant": "fraud_C",
                    "category": "travel",
                    "amt": "130.0",
                    "first": "B",
                    "last": "B",
                    "gender": "M",
                    "street": "x",
                    "city": "x",
                    "state": "WA",
                    "zip": "22222",
                    "lat": "45.0",
                    "long": "-120.0",
                    "city_pop": "900",
                    "job": "analyst",
                    "dob": "1985-01-01",
                    "trans_num": "t3",
                    "unix_time": "1546390800",
                    "merch_lat": "45.2",
                    "merch_long": "-120.2",
                    "is_fraud": "1",
                },
            ]
            rows_test = [
                {
                    "": "3",
                    "trans_date_trans_time": "2019-01-03 01:00:18",
                    "cc_num": "2222",
                    "merchant": "fraud_D",
                    "category": "entertainment",
                    "amt": "21.0",
                    "first": "B",
                    "last": "B",
                    "gender": "M",
                    "street": "x",
                    "city": "x",
                    "state": "WA",
                    "zip": "22222",
                    "lat": "45.0",
                    "long": "-120.0",
                    "city_pop": "900",
                    "job": "analyst",
                    "dob": "1985-01-01",
                    "trans_num": "t4",
                    "unix_time": "1546477200",
                    "merch_lat": "45.3",
                    "merch_long": "-120.3",
                    "is_fraud": "0",
                }
            ]
            self._write_csv(dataset_dir / "fraudTrain.csv", rows_train)
            self._write_csv(dataset_dir / "fraudTest.csv", rows_test)

            category_map = algo._load_sparkov_category_map(ROOT / "run_presets" / "sparkov_category_skill_map.json")
            profile = algo._derive_sparkov_profile(dataset_dir, category_map)

            self.assertEqual(profile["rows_profiled"], 4)
            self.assertEqual(profile["rows_mapped"], 4)
            self.assertAlmostEqual(profile["mapping_coverage"], 1.0)
            self.assertGreater(profile["fraud_rate"], 0.0)
            self.assertEqual(profile["unique_cards"], 2)
            self.assertEqual(profile["unique_merchants"], 4)

            context_weight_sum = sum(float(v) for v in profile["context_weights"].values())
            self.assertTrue(math.isclose(context_weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9))

            for context_name, skill_weights in profile["skill_weights"].items():
                total = sum(float(v) for v in skill_weights.values())
                self.assertTrue(math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9))

    def test_apply_recommendations_sets_missing_fields_only(self) -> None:
        args = argparse.Namespace(
            max_offers_per_agent=None,
            max_wants_per_agent=None,
            decomposition_rate=None,
            cross_context_want_rate=None,
        )
        profile = {
            "recommended": {
                "max_offers_per_agent": 3,
                "max_wants_per_agent": 4,
                "decomposition_rate": 0.42,
                "cross_context_want_rate": 0.18,
            }
        }
        algo._apply_sparkov_recommendations(args, profile)
        self.assertEqual(args.max_offers_per_agent, 3)
        self.assertEqual(args.max_wants_per_agent, 4)
        self.assertAlmostEqual(args.decomposition_rate, 0.42)
        self.assertAlmostEqual(args.cross_context_want_rate, 0.18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
