# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
import gym
import time
import sys
from helper import FrameStack
from replay_buffer import ReplayBuffer
import argparse
import readchar
from copy import deepcopy

def map(action):
    """ Maps the the given string to the corresponding
        action for the env (0-6)
        w -> 1 <-0
        s -> 4
        d -> 2
        a -> 3
        Args:
            param1(str): action
        Return:
            action in int
        
    """
    if action == "w":
        return 0
    if action == "s":
        return 3
    if action == "d":
        return 1
    if action == "a":
        return 2
    if action == "f":
        return 3


def main(args):
    """ Creates expert examples for Pacman env for Inverse Q Learning
    """
    print(sys.version)
    env = gym.make(args.env_name)
    print(env.observation_space)
    memory = ReplayBuffer((3, 84, 84), (1, ), args.buffer_size, 0, 0, args.device)
    if args.continue_samples:
        memory.load_memory(args.path)
        print("continue with {}  samples".format(memory.idx))
    env = FrameStack(env, args)
    state = env.reset()
    env.render()
    done = False
    steps = memory.idx
    score = 0
    jump = False
    while True:
        steps += 1
        print("memory idx {}".format(memory.idx))
        while True:
            try:
                input_action = readchar.readchar()
                if input_action == "r":
                    print("before memory idx {}".format(memory.idx))
                    memory.idx = memory.idx - 4
                    print("after memory idx {}".format(memory.idx))
                
                if input_action == "2":
                    snapshot2 = env.ale.cloneState()
                    print("save env to continue")
                
                if input_action == "3":
                    env.ale.restoreState(snapshot2)
                    print("load env to continue")
                
                if input_action == "1":
                    # saved_state = env.sim.get_state()
                    snapshot = env.ale.cloneState()
                    idx = memory.idx
                    print("save env")
                
                if input_action == "l":
                    #env.sim.set_state(saved_state)
                    env.ale.restoreState(snapshot)
                    memory.idx = idx
                    print("load env")
                
                if input_action == "m":
                    memory.save_memory("expert_policy-{}/".format(steps))

                if input_action == "f" or input_action ==" " or input_action == "q":
                    i = 0
                    jump = True
                
                if input_action == "p":
                    print("wanna save buffer press s")
                    io = readchar.readchar()
                    if io == "s":
                        memory.save_memory("expert_policy-{}/".format(steps))
                    sys.exit()
                    print("exit")
                    env.close()
                    return
                action = map(input_action)
                if action > 13:
                    continue
                break
            except Exception:
                continue
        if jump:
            while True:
                i +=1
                env.render()
                next_state, reward, done, _ = env.step(action)
                #memory.add(state, action, state, done)
                state = next_state
                if done:
                    break
                if i >= 80:
                    action = 0
                    jump =False
                    break
        next_state, reward, done, _ = env.step(action)
        env.render()
        if action > 4:
            sys.exit()
        memory.add(state, action, next_state, done)
        state = next_state
        print("action", action)
        print("reward", reward)
        score += reward
        if done:
            print("Episodes exit with score {}".format(score))
            break
    print("sampels in buffer ", memory.idx)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="MsPacman-v0", type=str)
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--buffer_size', default=100000, type=int)
    parser.add_argument('--continue_samples', default=False, type=bool)
    parser.add_argument('--path', default='expert_policy/', type=str)
    arg = parser.parse_args()
    main(arg)
