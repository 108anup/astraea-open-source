#!/usr/bin/env python3

import argparse
import sys
import os
from os import path
import numpy as np
import tensorflow as tf
import json
import signal
import math
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

import context
from agent.agent import Agent
from agent.definitions import transform_state, ACTION_DIM, STATE_DIM, GLOBAL_DIM
from helpers.utils import Params
from helpers.logger import logger
from helpers.ipc_socket import IPCSocket

config_path = path.abspath(
    path.join(path.dirname(__file__), os.pardir, "train", "astraea.json")
)
model_path = path.abspath(path.join(path.dirname(__file__), os.pardir, "model", "current"))

last_actions = [0] * 10


def map_action(action, cwnd):
    if action >= 0:
        out = 1 + 0.025 * (action)
        out = math.ceil(out * cwnd)
    else:
        out = 1 / (1 - 0.025 * (action))
        out = math.floor(out * cwnd)
    return out


def get_action_info():
    action_scale = np.array([1.0])
    action_range = (-action_scale, action_scale)
    return action_scale, action_range


def inference(flow_id, agent, state, s0_rec_buffer_inf=None):
    logger.info("inference start: flow_id: {}, state: {}".format(flow_id, state))
    s0, _ = transform_state(state)
    if s0_rec_buffer_inf is None:
        s0_rec_buffer_inf = np.zeros(agent.s_dim)
    s0_rec_buffer_inf = np.concatenate((s0_rec_buffer_inf[len(s0) :], s0))
    print(s0_rec_buffer_inf)
    a = agent.get_action(s0_rec_buffer_inf, False)
    a = a[0][0][0]
    action = map_action(a, state["cwnd"])
    # last_actions[flow_id] = a
    return action, s0_rec_buffer_inf, a


def make_agent(
    model_path, params, s_dim, s_dim_global, a_dim, action_scale, action_range
):
    agent = Agent(
        s_dim,
        s_dim_global,
        a_dim,
        batch_size=params.dict["batch_size"],
        h1_shape=params.dict["h1_shape"],
        h2_shape=params.dict["h2_shape"],
        stddev=0.05,
        policy_delay=params.dict["policy_delay"],
        mem_size=params.dict["memsize"],
        gamma=params.dict["gamma"],
        lr_c=params.dict["lr_c"],
        lr_a=params.dict["lr_a"],
        tau=params.dict["tau"],
        PER=params.dict["PER"],
        LOSS_TYPE=params.dict["LOSS_TYPE"],
        noise_type=3,
        noise_exp=params.dict["noise_exp"],
        train_exp=params.dict["train_exp"],
        action_scale=action_scale,
        action_range=action_range,
        is_global=params.dict["global"],
        ckpt_dir=model_path,
    )
    # init tf SingularMonitoredSession
    eval_sess = tf.Session()
    # load model
    agent.assign_sess(eval_sess)
    agent.load_model()
    return agent


def plot_cwnd_history(prefix):
    data = np.loadtxt("{}.log".format(prefix))
    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.title("CWND")
    plt.savefig("{}.png".format(prefix))
    sys.stderr.write("Astraea RL helper: ploted cwnd....\n")


def main():
    cwnd_history = []
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    global agent
    global params

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=config_path, help="configuration file"
    )
    parser.add_argument(
        "--model-path", type=str, default=model_path, help="path of saved models"
    )

    args = parser.parse_args()
    action_scale, action_range = get_action_info()
    params = Params(args.config)
    # world = Params(params.dict["world"])

    sys.stderr.write(f"PyHelper: Loading model from: {args.model_path}\n")

    s_dim, a_dim, s_dim_global = STATE_DIM, ACTION_DIM, GLOBAL_DIM
    single_dim = s_dim
    if params.dict["recurrent"]:
        s_dim = single_dim * params.dict["rec_dim"]
    agent = make_agent(
        args.model_path, params, s_dim, s_dim_global, a_dim, action_scale, action_range
    )
    s0_rec_buffer_inf = np.zeros(s_dim)

    state = {
        "avg_thr": 12 * 1e6,
        "max_tput": 48.8 * 1e6,
        "pacing_rate": 12.5 * 1e6,
        "cwnd": 150,
        "avg_urtt": 30.5 * 1000,
        "min_rtt": 30 * 1000,
        "srtt_us": 30.5 * 1000 * 8,
        "loss_ratio": 0,
        "packets_out": 150,
        "retrans_out": 0,
    }

    delay = 30
    # bin = 0.5
    bin = 1
    delay_upper_bound = 80

    fig, ax = plt.subplots()
    # for bw in [1.2, 2.4, 6, 7.2, 12]:
    for bw in [1.2, 4.8, 9.6, 14.4, 19.2]:
    # for bw in [1,2,3,4,5,6,7,8,9,10]:
    # for bw in [12]:
        delay_list = []
        action_list = []
        state["avg_thr"] = bw * 1e6
        # state["avg_thr"] = 5709365
        state["pacing_rate"] = bw * (1.25 / 1.2) * 1e6
        # state["pacing_rate"] = 6088291
        state["min_rtt"] = 31324
        state["max_tput"] = bw * (1.22 / 1.2) * 1e6
        # state["cwnd"] = (bw / 12) * 300
        # state["packets_out"] = (bw / 12) * 300

        while delay < delay_upper_bound:
            # logger.info("RL: state is {}".format(state))
            state["avg_urtt"] = delay * 1e3
            # state["avg_urtt"] = 36625
            state["srtt_us"] = delay * 1e3 * 8
            # state["srtt_us"] = 300100
            # state["loss_ratio"] = bw * 0.001 
            state["cwnd"] = 250 * (delay / 30) * (bw / 12)
            # state["cwnd"] = 156 
            state["packets_out"] = 250 * (delay / 30) * (bw / 12)
            # state["packets_out"] = 147
            
            _, s0_rec_buffer_inf, act = inference(
                -1, agent, state, s0_rec_buffer_inf=s0_rec_buffer_inf
            )
            # clear buffer
            delay += bin
            print(f"delay: {delay} - action: {act}")

            delay_list.append(delay)
            action_list.append(act)
        s0_rec_buffer_inf = np.zeros(s_dim)
        delay = 30

        plt.plot(delay_list, action_list, label="Throughput {} Mbps".format(100 * (bw / 12) ))
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Action")
    ax.legend()
    ax.grid(True)
    ax.minorticks_on()
    fig.savefig("astraea-contract.pdf")


if __name__ == "__main__":
    main()
