"""SMACv2 observation codec. Feature metadata is public; unit objects never enter rules."""
from __future__ import annotations
import re
import torch
from .environments import SchemaBuilder


def codec(feature_names, agents):
    builder = SchemaBuilder(agents)
    mappings = []
    for a in range(agents):
        names = list(feature_names)
        others = [i for i in range(agents) if i != a]
        for index, name in enumerate(names):
            match = re.match(r"(ally_.*?)(\d+)(_.*)?$", name)
            if match:
                # Upstream names are from agent 0's perspective: slot 1 is the
                # first teammate, not globally agent 1 for every observer.
                slot = int(match.group(2)) - 1
                names[index] = match.group(1) + str(others[slot]) + (match.group(3) or "")
        i, mapping = 0, []
        while i < len(names):
            name = names[i]
            if "unit_type" in name and "_bit_" in name:
                base = name.rsplit("_bit_", 1)[0]
                end = i + 1
                while end < len(names) and names[end].startswith(base + "_bit_"):
                    end += 1
                categories = ["unobserved"] + [f"type{j}" for j in range(end-i)]
                builder.add(f"agent{a}.{base}", a, categories)
                mapping.append((f"agent{a}.{base}", i, end, categories))
                i = end
            else:
                builder.add(f"agent{a}.{name}", a, tolerance=1e-5, unit="normalized_native")
                mapping.append((f"agent{a}.{name}", i, i+1, None))
                i += 1
        mappings.append(mapping)
    return builder.build(), mappings


def encode(schema, mappings, observations):
    values = {}
    for obs, mapping in zip(observations, mappings):
        for name, start, end, categories in mapping:
            if categories:
                v = torch.as_tensor(obs[start:end])
                values[name] = categories[int(v.argmax()) + 1] if v.sum() > 0 else "unobserved"
            else:
                values[name] = float(obs[start])
    return schema.encode(values)


def available_from_observation(schema, observation, action_size):
    """Native SMAC availability features, decoded from real or imagined input."""
    decoded=schema.decode(observation)
    masks=[]
    for agent in range(schema.shape[1]):
        prefix=f"agent{agent}."
        alive=decoded.get(prefix+"own_health",0)>0
        mask=[False]*action_size
        mask[0]=not alive
        if alive:
            mask[1]=True
            for index,direction in enumerate(("north","south","east","west"),2):
                mask[index]=decoded.get(prefix+"move_action_"+direction,0)>.5
            for index in range(6,action_size):
                mask[index]=decoded.get(prefix+f"enemy_shootable_{index-6}",0)>.5
        masks.append(mask)
    return masks
