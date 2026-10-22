"""Regression tests for the all-trades empirical observation rule.

Run: python -m unittest discover -s analysis/tests -v
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "code/figure2_trading_dynamics_zi_v2.py"
spec = importlib.util.spec_from_file_location("figure2_under_test", SOURCE)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def fixture(legs, b=1, coefficient=10, max_q=5):
    rows = []
    holdings = {1: 3, 2: 3}
    trades = []
    for index, (buyer, seller, quantity, price) in enumerate(legs, 1):
        holdings[buyer] += quantity
        holdings[seller] -= quantity
        trades.append(dict(session_code="test", round_raw=1, trade_index=index,
                           seconds=index, buyer_id=buyer, seller_id=seller,
                           quantity=quantity, price=price, eq_price=50))
    for pid in (1, 2):
        rows.append(dict(session_code="test", round_raw=1, id_in_group=pid,
                         permits=3, current_permits=holdings[pid],
                         group_cap_total=6, carbon_emission_per_unit=b,
                         max_production=max_q, marginal_cost_coefficient=coefficient,
                         market_price=100, disturbance_values=[0] * max_q))
    meta = pd.Series(dict(round=1, first=1, is_equal=1, allocation_method="test"))
    return model.reconstruct_one_market(pd.DataFrame(trades), pd.DataFrame(rows), meta)


class InclusionTests(unittest.TestCase):
    def assert_all_count(self, packages, units, summary):
        self.assertTrue(units.buyer_value_eligible.all())
        self.assertTrue(units.seller_cost_eligible.all())
        self.assertTrue(summary["reconstruction_ok"])
        self.assertEqual(summary["n_buyer_eligible_units"], len(units))
        self.assertEqual(summary["n_seller_eligible_units"], len(units))
        np.testing.assert_array_equal(packages.buyer_n_eligible_units, packages.quantity)
        np.testing.assert_array_equal(packages.seller_n_eligible_units, packages.quantity)

    def test_sell_then_buyback_then_new_purchase(self):
        p, u, _, summary = fixture([(2, 1, 1, 5), (1, 2, 1, 7), (1, 2, 1, 11)])
        self.assert_all_count(p, u, summary)
        self.assertEqual(u.iloc[0].seller_permit_cost, 70)
        self.assertEqual(u.iloc[1].buyer_permit_value, 70)  # h=2 -> 3 buyback
        self.assertEqual(u.iloc[2].buyer_permit_value, 60)  # h=3 -> 4
        self.assertEqual(list(u.buyer_permit_ordinal), [4, 3, 4])

    def test_purchase_then_resale_then_initial_sale(self):
        p, u, _, summary = fixture([(1, 2, 1, 11), (2, 1, 1, 13), (2, 1, 1, 17)])
        self.assert_all_count(p, u, summary)
        self.assertEqual(u.iloc[0].buyer_permit_value, 60)
        self.assertEqual(u.iloc[1].seller_permit_cost, 60)  # not purchase price 11
        self.assertEqual(u.iloc[2].seller_permit_cost, 70)
        self.assertEqual(list(u.seller_permit_ordinal), [3, 4, 3])

    def test_package_mean_and_tied_unit_order(self):
        p, u, _, summary = fixture([(2, 1, 1, 5), (1, 2, 2, 7)])
        self.assert_all_count(p, u, summary)
        self.assertEqual(p.iloc[1].buyer_permit_value_avg, 65)  # (70+60)/2
        self.assertEqual(p.iloc[1].buyer_permit_value_total, 130)
        self.assertEqual(list(u.transaction_order), [1, 2, 2])
        self.assertEqual(list(u.price), [5, 7, 7])

    def test_two_permit_averaging_unchanged(self):
        p, u, _, summary = fixture([(1, 2, 1, 11), (2, 1, 1, 13)], b=2,
                                    coefficient=20, max_q=3)
        self.assert_all_count(p, u, summary)
        self.assertEqual(u.iloc[0].buyer_permit_value, 30)
        self.assertEqual(u.iloc[1].seller_permit_cost, 30)
        self.assertNotEqual(u.iloc[1].seller_permit_cost, 11)

    def test_zero_marginal_value_is_still_an_observation(self):
        p, u, _, summary = fixture([(1, 2, 1, 11)], max_q=2)
        self.assert_all_count(p, u, summary)
        self.assertEqual(u.iloc[0].buyer_permit_value, 0)
        self.assertEqual(u.iloc[0].seller_permit_cost, 0)


if __name__ == "__main__":
    unittest.main()
