# -*- coding: utf-8 -*-
"""This script runs the agent-based Susceptible-Infected-Removed (SIR) Model.
Authors:
    Ying-Qiu Zheng, Shady Rahayel
    
For running the model, run:
    
    python abm.py --retro True --speed 10 --spreading-rate 0.01 --time 30000
        --delta-t 0.1 --seed -1 --seed-amount 1
        
--retro True specifies a retrograde spreading
--speed is the spreading speed of agents in edges
--spreading-rate is the probability of staying inside a region
--time is the spreading time of agents
--delta-t is the size of timesteps
--seed is an integer that refers to the list of regions listed
alphabetically from the Allen Mouse Brain Atlas (see params_nature_retro.pickle)
CP = 35, ACB = 3, and CA1 = 24
--seed-amount is the initial injected amount of infected agents
        
This generates arrays containing the number of normal and infected agents
at each iteration for every region of the Allen Mouse Brain Atlas.
The distribution of normal agents can be found in .s_region_history
The distribution of infected agents can be found in .i_region_history

"""

import sys
import numpy as np
import pickle
from AgentBasedModel import AgentBasedModel
from scipy.stats import zscore, norm
from tqdm import tqdm
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--retro", default=True, dest="retro",
        type=str2bool, nargs='?', help="Retrograde spreading (True) "
    )
    parser.add_argument(
        "-v", "--speed", default=10, dest="v", nargs='?',
        help="Spreading speed", type=float
    )
    parser.add_argument(
        "-s", "--spreading-rate", default=0.01, type=float,
        nargs='?', help="Spreading rate", dest="spread_rate"
    )
    parser.add_argument(
        "-t", "--time", default=30000, type=int, nargs='?',
        dest="total_time", help="Total spreading time"
    )
    parser.add_argument(
        "-d", "--delta-t", default=0.1, type=float, nargs='?',
        dest="dt", help="Size of time increment"
    )
    parser.add_argument(
        "-S", "--seed", default=-1, type=int, nargs='?',
        dest="seed", help="Simulated seeding site of misfolded alpha-syn"
    )
    parser.add_argument(
        "-a", "--seed-amount", default="1", type=float,
        dest="injection_amount", help="Seeding amount of misfolded alpha-syn"
    )
    parser.add_argument(
        "-c", "--clearance", default=None, dest="clearance_gene",
        help="Specify the gene modulating clearance"
    )
    args = parser.parse_args()
    return args


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_params(retro=False):
    # load synthesis gene
    # load snca
    with open('snca_nature.pickle', 'rb') as f:
        epr = pickle.load(f)
    
    syngene = norm.cdf(
        zscore(
            zscore(epr[0]) + zscore(epr[1])
        )
    )
        
    # load control genes
    # for APP, enter 'app_nature.pickle'
    # for MAPT, enter 'mapt_nature.pickle'
    # for NF1, enter 'nf1_nature.pickle'
    #with open('app_nature.pickle','rb') as f:
    #    epr = pickle.load(f)
    #    
    #syngene = norm.cdf(
    #        zscore(epr[0])
    #        )
    
    # load homogeneous values
    # choose the rate from 0.1 to 0.9
    #homorate = 0.5
    #syngene = np.transpose(np.full((1,213),homorate))

    if retro is False:
        with open('params_nature.pickle', "rb") as f:
            params = pickle.load(f)

        weights = params['sconn']
        distance = params['sdist']
        region_size = params['roisize']
        sources = params['source_list']
        targets = params['target_list']

        return (
            weights, distance, np.append(region_size, region_size),
            sources, targets, np.append(syngene, syngene)
        )

    elif retro is True:
        with open('params_nature_retro.pickle', "rb") as f:
            params = pickle.load(f)

        weights = params['sconn']
        distance = params['sdist']
        region_size = params['roisize']
        sources = params['source_list']
        targets = params['target_list']

        return (
            weights, distance, np.append(region_size, region_size),
            sources, targets, np.append(syngene[:len(sources)], syngene[:len(sources)])
        )


def load_clearance(clearance_gene=None):
    if clearance_gene is None:
        return 0.5 # default clearance rate
    else:
        with open("{}_nature.pickle".format(clearance_gene), 'rb') as f:
            epr = pickle.load(f)
    epr = np.append(epr, epr, axis=1)

    return norm.cdf(zscore(epr.flatten()))


if __name__ == "__main__":
    # run ABM
    # read arguments
    args = parse_arguments()

    retro = args.retro
    #injection_site = args.injection_site
    v = args.v
    spread_rate = args.spread_rate
    dt = args.dt
    seed = args.seed
    injection_amount = args.injection_amount
    total_time = args.total_time
    clearance_gene = args.clearance_gene
    weights, distance, region_size, sources, targets, syngene = load_params(retro=retro)
    clearance_rate = load_clearance(clearance_gene)
    abm = AgentBasedModel(
        weights=weights, distance=distance, region_size=region_size,
        sources=sources, targets=targets, dt=1
    )

    abm.set_growth_process(growth_rate=syngene)
    abm.set_clearance_process(clearance_rate=clearance_rate)
    abm.set_spread_process(v=v)
    abm.update_spread_process(spread_scale=spread_rate)

    # growth process
    print("Begin protein growth process....")
    for t in range(30000):
        prev = np.copy(abm.s_region)
        abm.growth_step()
        abm.clearance_step()
        abm.s_spread_step()
        if np.where(np.abs(prev - abm.s_region) / abm.s_region > 1e-7, 1, 0).sum() == 0:
            break
    abm.dt = 0.1
    for t in range(500000000):
        prev = np.copy(abm.s_region)
        abm.growth_step()
        abm.clearance_step()
        abm.s_spread_step()
        if np.where(np.abs(prev - abm.s_region) / abm.s_region > 1e-7, 1, 0).sum() == 0:
            break
    abm.dt = dt
    for t in range(1000000000):
        prev = np.copy(abm.s_region)
        abm.growth_step()
        abm.clearance_step()
        abm.s_spread_step()
        if np.where(np.abs(prev - abm.s_region) / abm.s_region > 1e-7, 1, 0).sum() == 0:
            break

    # spread process
    print("Begin protein spreading process...")
    print("Inject infectious proteins into {}...".format(abm.targets[seed]))
    abm.injection(seed=seed, amount=injection_amount)

    for t in tqdm(range(total_time)):
        abm.growth_step()
        abm.clearance_step()
        abm.trans_step()
        abm.s_spread_step()
        abm.i_spread_step()
        abm.record_history_region()
        abm.record_history_edge()

    abm_filename = "abm_spread_v.{0}.spread_rate.{1}.dt.{2}.seed.{3}.injection_amount.{4}.clearance_gene.{5}.pickle" \
        .format(v, spread_rate, dt, seed, injection_amount, clearance_gene)
    if retro is True:
        abm_filename = "retro_" + abm_filename

    with open(abm_filename, "wb") as f:
        pickle.dump(abm, f)
