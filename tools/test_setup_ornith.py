"""Ornith-1.5-122B (docs/ORNITH.md): an opt-in own-path family, registered only in a process started with --family ornith.
Runs without a GPU or downloads: python tools/test_setup_ornith.py"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(argv):
    old = sys.argv
    sys.argv = ["setup.py", *argv]
    try:
        spec = importlib.util.spec_from_file_location(f"setup_ornith_{len(argv)}", ROOT / "setup.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old


class Ornith(unittest.TestCase):
    def test_not_registered_unless_asked(self):
        s = load([])
        self.assertNotIn("ornith", s.OWN_PATH_FAMILIES)
        self.assertNotIn("unsloth/Ornith-1.5-122B-GGUF", s.REPO_PINS)

    def test_registered_with_family_flag(self):
        for argv in (["--family", "ornith"], ["--family=ornith"], ["launch", "--model", "iq3_xxs", "--family", "ornith"]):
            s = load(argv)
            self.assertIn("ornith", s.OWN_PATH_FAMILIES, argv)
        fam = s.OWN_PATH_FAMILIES["ornith"]
        self.assertEqual(fam["arch"], "qwen35moe")
        self.assertTrue(fam["local"])
        self.assertTrue(fam["split"])   # two cards: setup writes both and STRATA_Q36_SPLIT=auto
        for m in fam["models"]:   # every size names a file the table knows (the menu reads its size)
            self.assertIn(s.own_path_file(fam, m), fam["sha256"])
        self.assertEqual(fam["kv_bytes_per_token"], 13 * 2 * (256 + 4) * 2)   # int8 K and V with a scale per head
        self.assertEqual(s.file_pin(fam["mmproj_hf"] + fam["mmproj"]), fam["sha256"]["mmproj-F16.gguf"])
        self.assertEqual(s.own_path_size(fam, 62.0), "UD-IQ2_XXS")   # the family's pick: it fits two 24 GB cards
        self.assertEqual(s.own_path_file(fam, "UD-IQ2_XXS"), "Ornith-1.5-122B-UD-IQ2_XXS.gguf")
        self.assertEqual(s.own_path_bytes(fam, "UD-Q4_K_XL"), 10942929 + 49971099680 + 29731729056)   # all three shards
        self.assertEqual(s.own_path_bytes(fam, "UD-IQ2_XXS"), 41967395072)
        self.assertTrue(fam["kv8"])
        self.assertEqual(sorted(n for n in fam["sha256"] if n.startswith("Ornith-1.5-122B-UD-Q4_K_XL-")),
                         [f"Ornith-1.5-122B-UD-Q4_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)])


if __name__ == "__main__":
    unittest.main()
