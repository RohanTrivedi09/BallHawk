"""Write run.py from run_template.py with the current ballhawk_common.py and splits.json embedded.
usage: python3 kaggle/seed_replication/build.py   then   kaggle kernels push -p kaggle/seed_replication --accelerator NvidiaTeslaT4
"""
import pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
embed = {name: (ROOT / name).read_text() for name in ("ballhawk_common.py", "splits.json")}
(HERE / "run.py").write_text((HERE / "run_template.py").read_text().replace("{{EMBED}}", repr(embed)))
print("wrote", HERE / "run.py")
