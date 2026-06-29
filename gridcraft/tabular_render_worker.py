import os
import pickle
import struct
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "Gridcraft"))

from gridcraft import GridcraftConfig
from gridcraft.entities import AgentState, ItemDrop, MobState
from gridcraft.constants import Item
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
  config = GridcraftConfig(width=16, height=16, num_agents=1, view_size=7, tile_size=36)
  renderer = PygameRenderer(config)
  while True:
    message = read_message(sys.stdin.buffer)
    if message is None or message.get("cmd") == "close":
      break
    cmd = message.get("cmd")
    if cmd == "render_human":
      world = world_from_snapshot(message.get("world"))
      renderer.render(
        world,
        "human",
        tabular_observations=message["observation"],
        overlay_info=message.get("overlay_info"),
      )
      write_message(sys.stdout.buffer, {"ok": True})
    elif cmd == "poll_action":
      action, closed = poll_action(renderer, float(message.get("seconds", 0.1)))
      write_message(sys.stdout.buffer, {"action": action, "closed": closed})
    elif cmd == "wait":
      deadline = time.time() + float(message.get("seconds", 0.0))
      while time.time() < deadline:
        if renderer._pygame is not None:
          renderer._pump_events()
        time.sleep(0.05)
      write_message(sys.stdout.buffer, {"ok": True})
    else:
      world = world_from_snapshot(message.get("world"))
      frame = renderer.render(
        world,
        "rgb_array",
        tabular_observations=message["observation"],
        overlay_info=message.get("overlay_info"),
      )
      write_message(sys.stdout.buffer, frame)
  renderer.close()


def poll_action(renderer, seconds):
  pygame = renderer._pygame
  if pygame is None:
    return None, True

  key_to_action = {
    pygame.K_z: 1,
    pygame.K_s: 2,
    pygame.K_q: 3,
    pygame.K_d: 4,
  }
  deadline = time.time() + max(0.0, seconds)
  action = None
  closed = False
  while time.time() < deadline:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        renderer.close()
        closed = True
        break
      if event.type == pygame.KEYDOWN and event.key in key_to_action:
        action = key_to_action[event.key]
    if closed:
      break
    keys = pygame.key.get_pressed()
    for key, mapped_action in key_to_action.items():
      if keys[key]:
        action = mapped_action
    time.sleep(0.01)
  return action, closed


def world_from_snapshot(snapshot):
  if snapshot is None:
    return None

  class SnapshotWorld:
    def observations(self):
      return self._observations

  world = SnapshotWorld()
  world.terrain = snapshot["terrain"]
  world.blocks = snapshot["blocks"]
  world.items = [
    ItemDrop(
      item=Item(int(item["item"])),
      count=int(item.get("count", 1)),
      x=int(item["x"]),
      y=int(item["y"]),
    )
    for item in snapshot.get("items", [])
  ]
  world.mobs = [
    MobState(
      mob_id=int(mob.get("mob_id", i)),
      x=int(mob["x"]),
      y=int(mob["y"]),
      hp=int(mob.get("hp", 1)),
      alive=bool(mob.get("alive", True)),
    )
    for i, mob in enumerate(snapshot.get("mobs", []))
  ]
  world.agents = {}
  for agent_id, agent in snapshot.get("agents", {}).items():
    inventory = {
      Item(int(item_id)): int(count)
      for item_id, count in agent.get("inventory", {}).items()
    }
    inventory_order = [Item(int(item_id)) for item_id in agent.get("inventory_order", [])]
    equipped_value = agent.get("equipped")
    equipped = Item(int(equipped_value)) if equipped_value is not None else None
    world.agents[agent_id] = AgentState(
      agent_id=agent_id,
      x=int(agent["x"]),
      y=int(agent["y"]),
      hp=int(agent["hp"]),
      hunger=int(agent["hunger"]),
      inventory=inventory,
      inventory_order=inventory_order,
      equipped=equipped,
      alive=bool(agent.get("alive", True)),
    )
  world._observations = snapshot["observations"]
  return world


if __name__ == "__main__":
  main()
