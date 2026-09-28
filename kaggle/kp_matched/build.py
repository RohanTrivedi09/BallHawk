"""Write run.py with the current ballhawk_common.py, ballhawk_pitch.py and splits_pitch.json embedded.
usage: python3 kaggle/kp_matched/build.py   then   kaggle kernels push -p kaggle/kp_matched --accelerator NvidiaTeslaT4
"""
import pathlib
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
embed = {n: (ROOT / n).read_text() for n in ("ballhawk_common.py", "ballhawk_pitch.py", "splits_pitch.json")}
(HERE / "run.py").write_text((HERE / "run_template.py").read_text().replace("{{EMBED}}", repr(embed)))
print("wrote", HERE / "run.py")
