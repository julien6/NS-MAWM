from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import inspect
import json
import math
import random
from .rules import Assignment, Context, Fact, Library, RuleSpec, Tri, compile_rule, digest


def handcrafted(schema, rule_id, scope, reads, writes, effect, *, stage=0, guard=None, derives=None, assumptions=()):
    source = inspect.getsource(effect)
    reads = frozenset(reads)
    def applicability(c):
        if any((name[7:] not in c.memory if name.startswith("memory:") else name not in c.observation) for name in reads):
            return Tri.UNKNOWN
        return Tri.TRUE if effect(c) else Tri.FALSE
    return RuleSpec(rule_id, 0, stage, scope, reads, frozenset(writes),
                    guard or applicability, effect, derives,
                    assumptions=assumptions, code_hash=digest([Path(__file__).read_text(), source, [inspect.getsource(v) if callable(v) else v for v in (effect.__defaults__ or ())]]), code=source)


def gridcraft(schema, agents, view_size=7):
    rules = []
    radius = view_size // 2
    def key(a, kind, y, x):
        return f"agent{a}.{kind}[{y},{x}]"
    deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

    def displacement(c, a, joint=False):
        action = c.joint_action[a]
        if c.observation.get(f"agent{a}.health", 1) <= 0:
            return (0, 0)
        if action not in deltas:
            return (0, 0)
        dx, dy = deltas[action]
        y, x = radius + dy, radius + dx
        obs = c.observation
        terrain, block, entity = (obs.get(key(a, k, y, x)) for k in ("terrain", "block", "entity"))
        if terrain is None or block is None or entity is None:
            return None
        if terrain == "water" or block != "empty" or entity == "mob":
            return (0, 0)
        if entity == "agent":
            return (0, 0) if joint and all(i == a or act not in deltas for i, act in enumerate(c.joint_action)) else None
        # Earlier agents could enter this cell: individual knowledge cannot rule
        # it out. Joint stationary teammates establish the displacement exactly.
        if a > 0 and not (joint and all(c.joint_action[i] not in deltas for i in range(a))):
            return None
        return dx, dy

    def terrain_effect(c, a, rid, joint=False, water_only=False):
        action = c.joint_action[a]
        if water_only:
            if action not in deltas:
                return []
            dx, dy = deltas[action]
            if c.observation.get(key(a, "terrain", radius + dy, radius + dx)) != "water":
                return []
        d = displacement(c, a, joint)
        if d is None:
            return []
        dx, dy = d
        result = []
        for y in range(view_size):
            for x in range(view_size):
                sy, sx = y + dy, x + dx
                if 0 <= sy < view_size and 0 <= sx < view_size:
                    result.append(Assignment(key(a, "terrain", y, x), c.observation[key(a, "terrain", sy, sx)], rid, 0))
        return result

    for a in range(agents):
        own = [n for n in schema.by_name if n.startswith(f"agent{a}.")]
        terrain = [n for n in own if ".terrain[" in n]
        for name, scope in (("water", "individual"), ("terrain.shift", "individual"), ("collision", "joint")):
            rid = f"{name}.agent{a}"
            def effect(c, a=a, rid=rid, name=name):
                if name == "collision":
                    action = c.joint_action[a]
                    if action not in deltas:
                        return []
                    dx, dy = deltas[action]
                    if c.observation.get(key(a, "entity", radius + dy, radius + dx)) != "agent":
                        return []
                return terrain_effect(c, a, rid, joint=name == "collision", water_only=name == "water")
            rules.append(handcrafted(schema, rid, scope, own, terrain, effect,
                assumptions=("Sequential movement in agent index order; static terrain",)))
        rid = f"plank.agent{a}"
        def craft(c, a=a, rid=rid):
            p = f"agent{a}."
            if c.joint_action[a] != 9 or c.observation.get(p + "wood", 0) < 1 or c.observation.get(p + "health", 0) <= 0:
                return []
            return [Assignment(p + "wood", c.observation[p + "wood"] - 1, rid, 0),
                    Assignment(p + "plank", c.observation[p + "plank"] + 2, rid, 0)]
        rules.append(handcrafted(schema, rid, "individual", own, [f"agent{a}.wood", f"agent{a}.plank"], craft))
        rid = f"map.agent{a}"
        # A stationary observer can share an exactly aligned patch with a peer.
        # Alignment is established by a unique full terrain overlap, not hidden
        # coordinates; ambiguous homogeneous maps cause abstention.
        def shared_map(c, a=a, rid=rid):
            d = displacement(c, a, True)
            if d is None:
                return []
            dx, dy = d
            result = []
            for other in range(agents):
                if other == a:
                    continue
                offsets = []
                for oy in range(-radius, radius + 1):
                    for ox in range(-radius, radius + 1):
                        if ox == 0 and oy == 0:
                            continue
                        if c.observation.get(key(a, "entity", radius + oy, radius + ox)) != "agent":
                            continue
                        if c.observation.get(key(other, "entity", radius - oy, radius - ox)) != "agent":
                            continue
                        if all(c.observation[key(a, "terrain", y, x)] == c.observation[key(other, "terrain", y - oy, x - ox)]
                               for y in range(view_size) for x in range(view_size)
                               if 0 <= y - oy < view_size and 0 <= x - ox < view_size):
                            offsets.append((ox, oy))
                # Agent identity cannot be established with >2 agents from the
                # anonymous entity channel alone.
                if agents != 2 or len(offsets) != 1:
                    continue
                ox, oy = offsets[0]
                for y in range(view_size):
                    for x in range(view_size):
                        sx, sy = x + dx - ox, y + dy - oy
                        if 0 <= sx < view_size and 0 <= sy < view_size:
                            result.append(Assignment(key(a, "terrain", y, x), c.observation[key(other, "terrain", sy, sx)], rid, 0))
            return result
        rules.append(handcrafted(schema, rid, "joint", schema.by_name, terrain, shared_map,
            assumptions=("Two-agent unique alignment only; ambiguous identities abstain",)))
    return Library(rules, schema)


def overcooked(schema, agents=2, **kwargs):
    rules = []
    # Action.ALL_ACTIONS in the pinned benchmark: N,S,E,W,stay,interact.
    directions = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
    for a in range(agents):
        p = f"agent{a}."
        own = [n for n in schema.by_name if n.startswith(p)]
        def facing(c, p=p):
            x = int(c.observation[p + "x"] + c.observation[p + "dx"])
            y = int(c.observation[p + "y"] + c.observation[p + "dy"])
            return p + f"cell[{x},{y}]"
        for kind in ("pickup", "pot.add", "pot.timer", "deliver", "blocked.move", "player.collision"):
            rid = f"{kind}.agent{a}"
            def effect(c, a=a, p=p, rid=rid, kind=kind, facing=facing, own=tuple(own)):
                o, action = c.observation, c.joint_action[a]
                result = {}
                target = facing(c)
                terrain = o.get(target + ".terrain")
                held = o[p + "held"]
                if kind == "pickup" and action == 5 and held == "none" and terrain in ("O", "T", "D"):
                    result[p + "held"] = {"O": "onion", "T": "tomato", "D": "dish"}[terrain]
                elif kind == "deliver" and action == 5 and held == "soup" and terrain == "S":
                    result[p + "held"] = "none"
                elif kind == "pot.add" and action == 5 and held in ("onion", "tomato") and terrain == "P":
                    count = o[target + ".onions"] + o[target + ".tomatoes"]
                    # Another simultaneous interaction may fill/start the pot.
                    if count < 3 and o[target + ".cooking"] == 0 and o[target + ".ready"] == 0 and all(i == a or act != 5 for i, act in enumerate(c.joint_action)):
                        result[p + "held"] = "none"
                        result[target + "." + held + "s"] = o[target + "." + held + "s"] + 1
                elif kind == "pot.timer":
                    for name in own:
                        if name.endswith(".timer"):
                            base = name[:-6]
                            if o.get(base + ".terrain") == "P" and o[base + ".cooking"] == 1 and o[name] > 1:
                                result[name] = o[name] - 1
                elif kind in ("blocked.move", "player.collision") and action in directions:
                    dx, dy = directions[action]
                    x, y = int(o[p + "x"]), int(o[p + "y"])
                    nx, ny = x + dx, y + dy
                    blocked = o.get(p + f"cell[{nx},{ny}].terrain", "X") != " "
                    if kind == "player.collision":
                        q = f"agent{1-a}."
                        bx, by = int(o[q + "x"]), int(o[q + "y"])
                        ex, ey = directions.get(c.joint_action[1-a], (0, 0))
                        tx, ty = bx + ex, by + ey
                        if o.get(q + f"cell[{tx},{ty}].terrain", "X") != " ":
                            tx, ty = bx, by
                        blocked = (nx, ny) == (tx, ty) or ((nx, ny) == (bx, by) and (tx, ty) == (x, y))
                    if blocked:
                        result.update({p + "x": x, p + "y": y, p + "dx": dx, p + "dy": dy})
                return [Assignment(k, v, rid, 0) for k, v in result.items()]
            scope = "joint" if kind == "player.collision" else "individual"
            reads = schema.by_name if scope == "joint" else own
            rules.append(handcrafted(schema, rid, scope, reads, own, effect))
    return Library(rules, schema)


def predator_prey(schema, agents, **kwargs):
    rules = []
    def velocity(c, a):
        p = f"agent{a}."
        o = c.observation
        for name in o:
            if name.startswith(p) and name.endswith(".x") and any(s in name for s in ("landmark", "ally", "prey")):
                if math.hypot(o[name], o[name[:-1] + "y"]) < .5:
                    return None
        fx, fy = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[c.joint_action[a]]
        vx, vy = .75 * o[p + "vx"] + .3 * fx, .75 * o[p + "vy"] + .3 * fy
        speed = math.hypot(vx, vy)
        return (vx / max(speed, 1), vy / max(speed, 1))
    for a in range(agents):
        own = [n for n in schema.by_name if n.startswith(f"agent{a}.")]
        for kind in ("self.vel", "landmark.rel", "ally.rel"):
            rid = f"{kind}.agent{a}"
            def effect(c, a=a, rid=rid, kind=kind, own=tuple(own)):
                v = velocity(c, a)
                if v is None:
                    return []
                p = f"agent{a}."
                result = {}
                if kind == "self.vel":
                    result = {p + "vx": v[0], p + "vy": v[1]}
                elif kind == "landmark.rel":
                    result = {n: c.observation[n] - .1 * v[0 if n.endswith(".x") else 1]
                              for n in own if ".landmark" in n}
                else:
                    for b in range(agents):
                        if a == b:
                            continue
                        vb = velocity(c, b)
                        if vb is not None:
                            for axis, i in (("x", 0), ("y", 1)):
                                name = p + f"ally{b}.{axis}"
                                result[name] = c.observation[name] + .1 * (vb[i] - v[i])
                return [Assignment(k, value, rid, 0) for k, value in result.items()]
            rules.append(handcrafted(schema, rid, "joint" if kind == "ally.rel" else "individual",
                schema.by_name if kind == "ally.rel" else own, own, effect))
    return Library(rules, schema)


def smacv2(schema, agents, **kwargs):
    rules = []
    for a in range(agents):
        p = f"agent{a}."
        names = [n for n in schema.by_name if n.startswith(p)]
        def safe_alive(c, a):
            p = f"agent{a}."
            o = c.observation
            if o.get(p + "own_health", 0) <= 0:
                return False
            distances = [v for n, v in o.items() if n.startswith(p + "enemy_distance_")]
            # A full-health marine survives at most one six-damage shot per
            # enemy during eight game loops (6/45 < .14).
            return bool(distances) and o[p + "own_health"] > .14 * len(distances)
        for kind in ("dead.agent", "unit.type", "health.static", "ally.visibility"):
            rid = f"{kind}.agent{a}"
            def effect(c, a=a, p=p, names=tuple(names), kind=kind, rid=rid):
                o, result = c.observation, {}
                if kind == "dead.agent" and o.get(p + "own_health", -1) == 0:
                    result = {n: ("unobserved" if schema.by_name[n].kind == "categorical" else 0.) for n in names}
                elif kind == "unit.type" and safe_alive(c, a):
                    name = p + "own_unit_type"
                    if name in o and o[name] != "unobserved":
                        result[name] = o[name]
                elif kind == "health.static":
                    no_attacks = all(action < 6 for action in c.joint_action) and (c.prev_joint_action is None or all(action < 6 for action in c.prev_joint_action))
                    if no_attacks and safe_alive(c, a):
                        for n in names:
                            if "enemy_health_" in n:
                                index = n.rsplit("_", 1)[1]
                                distance = o.get(p + "enemy_distance_" + index, 0)
                                if 0 < distance < 1 - 4/9:
                                    result[n] = o[n]
                elif kind == "ally.visibility":
                    for b in range(agents):
                        if b == a:
                            continue
                        name = p + f"ally_visible_{b}"
                        if name not in schema.by_name:
                            continue
                        if o.get(p + "own_health", -1) == 0 or o.get(f"agent{b}.own_health", -1) == 0:
                            result[name] = 0.
                        elif safe_alive(c, a) and safe_alive(c, b):
                            distance = o.get(p + f"ally_distance_{b}", 0)
                            if 0 < distance < 1 - 4/9:
                                result[name] = 1.
                return [Assignment(n, v, rid, 0) for n, v in result.items()]
            scope = "joint" if kind in ("health.static", "ally.visibility") else "individual"
            rules.append(handcrafted(schema, rid, scope, schema.by_name if scope == "joint" else names, names, effect,
                assumptions=("Terran marine-only scenario; no regeneration or projectiles; sight=9; conservative displacement bound=2 per unit",)))
    return Library(rules, schema)


def make_library(env, schema=None):
    from .extensions import ENVIRONMENT_LIBRARIES
    factory = ENVIRONMENT_LIBRARIES.get(env.metadata["environment"])
    if factory is not None:
        return factory(env, schema or env.schema)
    return globals()[env.metadata["environment"]](schema or env.schema, env.agents,
        **({"view_size": env.view_size} if hasattr(env, "view_size") else {}))


def save_library(library, path):
    records = []
    for r in library.rules:
        records.append({k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in vars(r).items()
                        if k not in ("guard", "effect", "derives")})
    Path(path).write_text(json.dumps({"manifest": library.manifest(), "library_hash": library.hash, "rules": records}, indent=2))


def builtin_library(schema):
    names = schema.by_name
    if any(".terrain[" in n for n in names):
        size = int(round(sum(n.startswith("agent0.terrain[") for n in names) ** .5))
        return gridcraft(schema, schema.shape[1], size)
    if any(".cell[" in n for n in names):
        return overcooked(schema, schema.shape[1])
    if "agent0.prey.x" in names:
        return predator_prey(schema, schema.shape[1])
    if "agent0.own_health" in names:
        return smacv2(schema, schema.shape[1])
    return Library([], schema)


def load_library(path, schema):
    obj = json.loads(Path(path).read_text())
    builtins = {r.id: r for r in builtin_library(schema).rules}
    m = obj.get("manifest", {})
    if str(m.get("creator", "")).startswith("corruption:"):
        _, kind, fraction, seed = m["creator"].split(":")
        base_filter = m.get("filter", {})
        baseline = builtin_library(schema).filtered(**base_filter)
        restored = corrupt(baseline, kind, float(fraction), int(seed))
        if restored.hash != obj["library_hash"]:
            raise ValueError("Corruption provenance mismatch")
        return restored
    rules = []
    for record in obj["rules"]:
        if record["source"] == "handcrafted" and record["id"] in builtins:
            r = builtins.get(record["id"])
            if r is None or r.version != record["version"] or r.code_hash != record["code_hash"]:
                raise ValueError("Handcrafted rule source/version differs from this code checkout")
            rules.append(replace(r, test_cases=tuple(record.get("test_cases", ()))))
        else:
            rules.append(compile_rule(record))
    m = obj.get("manifest", {})
    library = Library(rules, schema, edges=m.get("edges", ()), policy=m.get("policy", "block_rejection"),
                      parent=m.get("parent"), creator=m.get("creator", "human"), rule_filter=m.get("filter"))
    if obj.get("library_hash") and obj["library_hash"] != library.hash:
        raise ValueError("Library content hash mismatch")
    return library


def corrupt(library, kind, fraction, seed):
    if kind not in ("random_values", "wrong_guards", "overclaimed_masks") or not 0 <= fraction <= .4:
        raise ValueError("Invalid corruption condition")
    rng = random.Random(seed)
    selected = set(rng.sample([r.id for r in library.rules], round(len(library.rules) * fraction)))
    rules = []
    for r in library.rules:
        if r.id not in selected:
            rules.append(r)
            continue
        version = r.version + 1
        def effect(c, r=r, version=version):
            evaluation_context = c
            if kind == "wrong_guards" and r.source == "handcrafted":
                # Deliberately evaluate the applicability premise for an action
                # other than the recorded action (a wrong action guard).
                evaluation_context = replace(c, joint_action=tuple(0 if a else 1 for a in c.joint_action))
            out = r.effect(evaluation_context)
            if kind == "overclaimed_masks":
                assigned = {a.block for a in out}
                out += [Assignment(n, v, r.id, r.version) for n, v in c.observation.items() if n not in assigned]
            changed = []
            for a in out:
                v = a.value
                if kind == "random_values":
                    b = library.schema.by_name[a.block]
                    local = random.Random(digest([seed, r.id, a.block, c.step]))
                    v = local.choice(b.categories) if b.kind == "categorical" else float(v) + local.uniform(-3, 3)
                changed.append(Assignment(a.block, v, r.id, version))
            return changed
        rules.append(replace(r, version=version, base_version=r.version, effect=effect,
            guard=(lambda c: Tri.TRUE) if kind == "wrong_guards" else r.guard,
            writes=frozenset(library.schema.by_name) if kind == "overclaimed_masks" else r.writes,
            reads=frozenset(library.schema.by_name) if kind == "overclaimed_masks" else r.reads,
            code_hash=digest([r.code_hash, kind, fraction, seed])))
    return Library(rules, library.schema, edges=library.edges, parent=library.hash,
                   creator=f"corruption:{kind}:{fraction}:{seed}", rule_filter=library.rule_filter)
