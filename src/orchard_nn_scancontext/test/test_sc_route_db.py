import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orchard_nn_scancontext.sc_route_db as scdb


class TestScRouteDb(unittest.TestCase):
    def _make_db(self, prototypes: np.ndarray, counts: np.ndarray) -> scdb.RouteDb:
        params = scdb.ScanContextParams(
            num_ring=int(prototypes.shape[1]),
            num_sector=int(prototypes.shape[2]),
            min_range=0.1,
            max_range=80.0,
            height_lower_bound=-1.0,
            height_upper_bound=9.0,
        )
        flat = prototypes.reshape(prototypes.shape[0], -1).astype(np.float32, copy=False)
        proto_norm_flat = scdb._normalize_rows(flat)
        return scdb.RouteDb(
            prototypes=prototypes.astype(np.float32, copy=False),
            proto_norm_flat=proto_norm_flat.astype(np.float32, copy=False),
            counts=counts.astype(np.int32, copy=False),
            params=params,
            meta={},
        )

    def test_cosine_masks_empty_segments(self) -> None:
        prototypes = np.zeros((3, 2, 2), dtype=np.float32)
        prototypes[1] = np.asarray([[-1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        prototypes[2] = np.asarray([[0.0, -1.0], [0.0, 0.0]], dtype=np.float32)
        counts = np.asarray([0, 1, 1], dtype=np.int32)
        db = self._make_db(prototypes, counts)

        desc = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        rid, conf = scdb.predict_route_id_cosine(desc, db, temperature=0.02)
        self.assertEqual(rid, 2)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_l2_masks_empty_segments(self) -> None:
        prototypes = np.zeros((3, 2, 2), dtype=np.float32)
        prototypes[1] = 1.0
        prototypes[2] = 2.0
        counts = np.asarray([0, 1, 1], dtype=np.int32)
        db = self._make_db(prototypes, counts)

        desc = np.zeros((2, 2), dtype=np.float32)
        rid, conf = scdb.predict_route_id_l2(desc, db, temperature=0.05)
        self.assertEqual(rid, 1)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_all_empty_returns_unknown(self) -> None:
        prototypes = np.zeros((2, 2, 2), dtype=np.float32)
        counts = np.asarray([0, 0], dtype=np.int32)
        db = self._make_db(prototypes, counts)

        desc = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(scdb.predict_route_id_cosine(desc, db), (-1, 0.0))
        self.assertEqual(scdb.predict_route_id_l2(desc, db), (-1, 0.0))


if __name__ == "__main__":
    unittest.main()
