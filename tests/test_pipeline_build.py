import unittest
from pathlib import Path

from build_pipeline import generated


class PipelineBuildTests(unittest.TestCase):
    def test_only_kernel_configuration_changes(self):
        base=(Path(__file__).resolve().parents[1]/"native/bridge.cu").read_text(encoding="utf-8")
        marker="// Original Requant Network experiment harness;"
        for tile in (128,256):
            for stages in ("auto","2","3","4"):
                for schedule in ("coop","ping"):
                    source=generated(base,tile,stages,schedule)
                    self.assertEqual(source.split(marker)[1],base.split(marker)[1])
                    self.assertEqual(source[:source.index("// Kernel Perf config")],base[:base.index("// Kernel Perf config")])
                    self.assertIn(f"Shape<_{tile},_128,_128>",source)
                    self.assertIn("PingpongNvf4Sm120" if schedule=="ping" else "KernelTmaWarpSpecializedNvf4Sm120",source)
        with self.assertRaises(ValueError):generated(base+base,128,"2","ping")


if __name__=="__main__":unittest.main()
