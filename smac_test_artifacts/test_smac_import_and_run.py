import os
import subprocess
import sys

sc2path = os.environ.get("SC2PATH", "")
print("SC2PATH =", sc2path)

# SMAC official smoke test (module): smac.examples.random_agents :contentReference[oaicite:8]{index=8}
cmd = [sys.executable, "-m", "smac.examples.random_agents"]
print("Running:", " ".join(cmd))
p = subprocess.run(cmd, check=False)
sys.exit(p.returncode)
