import json
import os


def append_progress(path, metrics, step=None, namespace=None):
  if not path:
    return
  directory = os.path.dirname(path)
  if directory:
    os.makedirs(directory, exist_ok=True)
  payload = {
    "metrics": metrics,
    "step": step,
    "namespace": namespace,
  }
  with open(path, "a") as f:
    f.write(json.dumps(payload) + "\n")


def read_progress(path, offset=0):
  if not path or not os.path.exists(path):
    return offset, []
  events = []
  with open(path) as f:
    f.seek(offset)
    for line in f:
      line = line.strip()
      if line:
        events.append(json.loads(line))
    offset = f.tell()
  return offset, events
