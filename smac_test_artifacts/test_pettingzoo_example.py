import os
import glob
import subprocess
import sys

smac_dir = os.environ.get("SMAC_DIR", "./smac")
pz_dir = os.path.join(smac_dir, "smac", "examples", "pettingzoo")
print("Looking for PettingZoo examples in:", pz_dir)

candidates = sorted(glob.glob(os.path.join(pz_dir, "*.py")))
if not candidates:
    print("No .py files found in smac/examples/pettingzoo.")
    print("Your repo version may not include PettingZoo examples, or paths changed.")
    sys.exit(0)

# Heuristic: prefer demo/test-like names if present
preferred = [c for c in candidates if any(k in os.path.basename(c).lower() for k in ["demo", "example", "test", "run"])]
target = preferred[0] if preferred else candidates[0]

print("Running PettingZoo example:", target)
p = subprocess.run([sys.executable, target], check=False)
sys.exit(p.returncode)
