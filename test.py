from replay_buffer import ReplayBuffer
import json


with open ("param.json", "r") as f:
    config = json.load(f)


memory = ReplayBuffer((config["history_length"], config["size"], config["size"]), (1,), config["buffer_size"],  config["device"])
memory.load_memory(config["buffer_path"])


a = 0
for idx in range(memory.idx):
    print(memory.actions[idx])
    if memory.actions[idx] > a :
        a=  memory.actions[idx]
print("max", a)
