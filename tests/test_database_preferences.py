import os
import tempfile
import unittest

from core.database import ScoreDB


class TestScorePreferences(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = ScoreDB(os.path.join(self.temp_dir.name, "scores.db"))
        self.score_id = self.db.add_score(
            "测试",
            [{"notes": ["mid_1"], "dur": 1.0}],
            bpm_default=100,
        )

    def tearDown(self):
        self.db.conn.close()
        self.temp_dir.cleanup()

    def test_preferences_are_isolated_by_profile(self):
        self.db.save_score_preferences(
            self.score_id,
            "delta_force_harmonica",
            {"version": 1, "transpose": 2, "segment": [1, 8]},
        )
        self.assertEqual(
            self.db.get_score_preferences(self.score_id, "delta_force_harmonica"),
            {"version": 1, "transpose": 2, "segment": [1, 8]},
        )
        self.assertEqual(self.db.get_score_preferences(self.score_id, "default"), {})

    def test_upsert_replaces_previous_payload(self):
        self.db.save_score_preferences(self.score_id, "default", {"version": 1, "bpm": 90})
        self.db.save_score_preferences(self.score_id, "default", {"version": 1, "bpm": 120})
        self.assertEqual(
            self.db.get_score_preferences(self.score_id, "default"),
            {"version": 1, "bpm": 120},
        )

    def test_deleting_score_cascades_preferences(self):
        self.db.save_score_preferences(self.score_id, "default", {"version": 1})
        self.db.delete_score(self.score_id)
        count = self.db.conn.execute(
            "SELECT COUNT(*) FROM score_preferences WHERE score_id=?", (self.score_id,)
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_rejects_non_object_preferences(self):
        with self.assertRaises(ValueError):
            self.db.save_score_preferences(self.score_id, "default", ["bad"])

    def test_batch_import_is_validated_before_any_insert(self):
        before = len(self.db.list_scores())
        records = [
            {"name": "有效", "notes": [{"notes": ["mid_1"], "dur": 1}], "bpm_default": 100},
            {"name": "无效", "notes": [{"notes": ["bad"], "dur": 1}], "bpm_default": 100},
        ]
        with self.assertRaises(ValueError):
            self.db.add_scores(records)
        self.assertEqual(len(self.db.list_scores()), before)

    def test_batch_import_commits_every_valid_score(self):
        ids = self.db.add_scores(
            [
                {"name": "一", "notes": [{"notes": ["mid_1"], "dur": 1}]},
                {"name": "二", "notes": [{"notes": ["mid_2"], "dur": 1}], "bpm_default": 120},
            ]
        )
        self.assertEqual(len(ids), 2)
        self.assertEqual([self.db.get_score(score_id)["name"] for score_id in ids], ["一", "二"])


if __name__ == "__main__":
    unittest.main()
