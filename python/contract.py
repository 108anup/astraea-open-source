#!/usr/bin/env python3

import argparse
import json
import math
import os
import signal
import sys
from os import path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import interpolate

matplotlib.use("Agg")
import time

import context
from agent.agent import Agent
from agent.definitions import ACTION_DIM, GLOBAL_DIM, STATE_DIM, transform_state
from helpers.ipc_socket import IPCSocket
from helpers.logger import logger
from helpers.utils import Params
from matplotlib import pyplot as plt

from plot_config_light import get_fig_size_paper, get_fig_size_ppt, get_style, get_fig_size_acm_small


style = get_style(False, True, True)  # paper
get_fig_size = get_fig_size_acm_small
matplotlib.rcParams.update(style)

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
        "packets_out": 150,
        "loss_ratio": 0,
        "retrans_out": 0,
    }

    bin = 1
    delay_lower_bound = 0
    delay_upper_bound = 25

    # For 100 Mbps with 2 flows and 30 ms RTprop.
    ref_bw_mbps = 100/2.0
    ref_state = {
        "avg_thr": 6115578,  # basically bytes per second
        "avg_urtt": 38663,
        "cnt": 31,
        "cwnd": 165,
        "loss_bytes": 0,
        "loss_ratio": 0.0,
        "max_packets_out": 164,
        "max_tput": 6340841,
        "min_rtt": 30279,
        "mss_cache": 1448,
        "pacing_rate": 6176097,
        "packets_out": 163,
        "retrans_out": 0,
        "snd_ssthresh": 2147483647,
        "srtt_us": 309477,
        "thr_cnt": 31,
        "time_delta": 29998,
    }

    records = []
    figsize = get_fig_size(0.32, 0.4)
    fig, ax = plt.subplots(figsize=figsize)
    # for bw_mbps in [10, 20, 30, 40, 50, 60, 70, 80]:
    for bw_mbps in [10, 30, 50, 70]:
        bw_ppms = bw_mbps/12.0
        delay_list = []
        action_list = []


        state["avg_thr"] = ref_state["avg_thr"] * (bw_mbps / ref_bw_mbps)
        state["pacing_rate"] = ref_state["pacing_rate"] * (bw_mbps / ref_bw_mbps)
        state["min_rtt"] = ref_state["min_rtt"]
        state["max_tput"] = ref_state["max_tput"] * (bw_mbps / ref_bw_mbps)

        # state["avg_thr"] = 5709365
        # state["pacing_rate"] = 6088291
        # state["cwnd"] = 156
        # state["avg_urtt"] = 36625
        # state["min_rtt"] = 31324
        # state["srtt_us"] = 300100
        # state["packets_out"] = 147
        # state["loss_ratio"] = 0
        # state["retrans_out"] = 0

        for delay_ms in range(delay_lower_bound, delay_upper_bound + 1, bin):
            rtt_ms = delay_ms + ref_state["min_rtt"] / 1000.0
            state["avg_urtt"] = rtt_ms * 1e3
            state["srtt_us"] = rtt_ms * 1e3 * 8  # as srtt is << 3
            state["cwnd"] = bw_ppms * rtt_ms
            state["packets_out"] = bw_ppms * rtt_ms

            _, s0_rec_buffer_inf, act = inference(
                -1, agent, state, s0_rec_buffer_inf=s0_rec_buffer_inf
            )
            # clear buffer
            print(f"delay: {delay_ms} - action: {act}")

            delay_list.append(delay_ms)
            action_list.append(act)
        s0_rec_buffer_inf = np.zeros(s_dim)

        f = interpolate.interp1d(
            action_list, delay_list, kind="linear", fill_value="extrapolate"
        )
        record = {
            "delay": f(0),
            "rate": bw_mbps,
        }
        records.append(record)

        ax.plot(delay_list, action_list, label="{}".format(bw_mbps))

    xx = np.linspace(delay_lower_bound, delay_upper_bound, 1000)
    yy = xx * 0
    ax.plot(xx, yy, color="black", linestyle="--", label="_")
    # ax.set_xlabel("Delay (ms)")
    # ax.set_ylabel("Action")
    ax.set_xlabel("Delay (ms)", labelpad=3)
    ax.set_ylabel("Action", labelpad=0)
    # ax.legend(ncol=2,)
    ax.legend()
    ax.grid(True)
    ax.minorticks_on()
    # fig.subplots_adjust(left=0.21)
    fig.tight_layout(pad=0.03)
    fig.savefig("astraea-contract2.pdf")

    df = pd.DataFrame(records)
    df.to_csv("astraea-contract2.csv", index=False)


if __name__ == "__main__":
    main()
