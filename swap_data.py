import json
from utils import mkdir
from replay_buffer import ReplayBuffer


with open ("param.json", "r") as f:
    param = json.load(f)

config = param
print(config["expert_buffer_size"])
memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["buffer_size"], config["image_pad"], config["seed"], config["device"])
memory.load_memory(config["buffer_path"])

memory_expert = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["seed"], config["device"])
print("buffer_size target", memory_expert.capacity)

for idx in range(memory.idx):
        memory_expert.add(memory.obses[idx], memory.actions[idx], memory.rewards[idx], memory.next_obses[idx], memory.not_dones[idx], memory.not_dones_no_max[idx])
print(memory_expert.idx)
path = "expert_policy-20k"
mkdir("", path)
memory_expert.save_memory(path)
