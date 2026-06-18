import os
import pickle
import struct
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "Gridcraft"))

from gridcraft import GridcraftConfig
from gridcraft.render import PygameRenderer


def read_message(stream):
  header = stream.read(4)
  if not header:
    return None
  size = struct.unpack("!I", header)[0]
  payload = stream.read(size)
  if len(payload) != size:
    return None
  return pickle.loads(payload)


def write_message(stream, message):
  payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
  stream.write(struct.pack("!I", len(payload)))
  stream.write(payload)
  stream.flush()


def main():
  config = GridcraftConfig(width=12, height=12, num_agents=1, view_size=7, tile_size=48)
  renderer = PygameRenderer(config)
  while True:
    message = read_message(sys.stdin.buffer)
    if message is None or message.get("cmd") == "close":
      break
    cmd = message.get("cmd")
    if cmd == "render_human":
      renderer.render(None, "human", tabular_observations=message["observation"])
      write_message(sys.stdout.buffer, {"ok": True})
    elif cmd == "wait":
      deadline = time.time() + float(message.get("seconds", 0.0))
      while time.time() < deadline:
        if renderer._pygame is not None:
          renderer._pump_events()
        time.sleep(0.05)
      write_message(sys.stdout.buffer, {"ok": True})
    else:
      frame = renderer.render(None, "rgb_array", tabular_observations=message["observation"])
      write_message(sys.stdout.buffer, frame)
  renderer.close()


if __name__ == "__main__":
  main()
