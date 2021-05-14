# -*- coding: utf-8 -*-
"""
Agent-based model for alpha-syn spreading on mouse connectome

"""

import numpy as np

class AgentBasedModel:
    def __init__(
            self, weights, distance,
            sources, targets, region_size, dt=0.1
    ):
        """
        construct the object. Note all the params are those that are fixed
        we don't need to fit them in model-fitting stage
        :param weights: ndarray-like, row-source column-target
        :param distance: ndarray-like, same as above; the index must match
        :param sources: list, labels of source regions
        :param targets: list, labels of target regions
        :param region_size: ndarray-like
        :param dt: float
        """
        self.adj = np.where(weights != 0, 1, 0)
        self.weights = np.copy(weights)
        self.distance = np.copy(distance)

        # region counts and size
        self.sources = sources
        self.n_sources = len(sources)
        self.targets = targets
        self.n_targets = len(targets)
        self.region_size = np.array(region_size)

        # spread rates
        self.spread_weights = self.weights / \
            self.weights.sum(axis=1)[:, np.newaxis]
        # index of diagonal matrix -- may be useful
        self.diagonal = np.eye(self.n_sources, self.n_targets)
        self.region_to_edge_weights = np.copy(self.spread_weights)
        self.region_to_edge_weights[self.diagonal == 1] = 0

        self.adj_dist = self.adj * self.distance
        self.dist_inv = np.zeros(self.distance.shape)
        self.dist_inv[self.adj_dist != 0] = 1 / \
            self.adj_dist[self.adj_dist != 0]
        self.edge_to_region_weights = np.copy(self.dist_inv)

        self.dt = dt

        # rates
        self.growth_rate = 0
        self.clearance_rate = 0.5
        self.trans_rate = 1 / self.region_size

        # initialize population
        self.s_region = np.zeros(
            (max(self.n_targets, self.n_sources), )
        )
        self.i_region = np.zeros(
            (max(self.n_targets, self.n_sources), )
        )
        self.s_edge = np.zeros(self.adj.shape)
        self.i_edge = np.zeros(self.adj.shape)

        # record history
        self.s_region_history = np.empty(
            (0, max(self.n_targets, self.n_sources))
        )
        self.i_region_history = np.empty(
            (0, max(self.n_targets, self.n_sources))
        )

        self.s_edge_history = np.empty(
            (0, max(self.n_targets, self.n_sources))
        )
        self.i_edge_history = np.empty(
            (0, max(self.n_targets, self.n_sources))
        )

    def set_spread_process(self, v):
        """v spread"""
        self.edge_to_region_weights *= v

    def set_growth_process(self, growth_rate):
        """growth_rate involves params to be fitted"""
        self.growth_rate = np.array(growth_rate)

    def set_clearance_process(self, clearance_rate):
        """clearance_rate involves params to be fitted"""
        self.clearance_rate = np.array(clearance_rate)

    def set_trans_process(self, trans_rate):
        """trans_rate involves params to be fitted"""
        self.trans_rate *= trans_rate

    def update_spread_process(self, v_scale=1, spread_scale=1):
        """
        slow_down, exit_down involve params to be fitted
        :param v_scale: float, by which the spreading in edges is discounted
        :param spread_scale: float, by which the probability of leaving regions
            is discounted
        """
        self.edge_to_region_weights *= v_scale
        self.region_to_edge_weights *= spread_scale

    def update_growth_process(self, growth_scale=1):
        """growth_down involves params to be fitted"""
        self.growth_rate *= growth_scale

    def update_clearance_process(self, clearance_scale=1):
        """clearance_down involves params to be fitted"""
        self.clearance_rate *= clearance_scale

    def update_trans_process(self, trans_scale=1):
        """trans_down involves params to be fitted"""
        self.trans_rate *= trans_scale

    def s_spread_step(self):
        """spread step in each time step"""
        region_to_edge = self.region_to_edge_weights * self.dt * \
            self.s_region[:self.n_sources][:, np.newaxis]

        edge_to_region = self.edge_to_region_weights * \
            self.s_edge * self.dt

        # update edges and regions
        self.s_edge += region_to_edge - edge_to_region

        self.s_region += edge_to_region.sum(axis=0) - \
            np.append(
                region_to_edge.sum(axis=1),
                np.zeros(
                    np.abs(self.n_targets - self.n_sources),
                ),
                axis=0
            )

    def i_spread_step(self):
        """spread step in each time step"""
        region_to_edge = self.region_to_edge_weights * self.dt * \
            self.i_region[:self.n_sources][:, np.newaxis]

        edge_to_region = self.edge_to_region_weights * \
            self.i_edge * self.dt

        # update edges and regions
        self.i_edge += region_to_edge - edge_to_region

        self.i_region += edge_to_region.sum(axis=0) - \
            np.append(
                region_to_edge.sum(axis=1),
                np.zeros(
                    np.abs(self.n_targets - self.n_sources),
                ),
                axis=0
            )

    def growth_step(self):
        self.s_region += self.growth_rate * self.dt * \
            self.region_size

    def clearance_step(self):
        """clearance step"""
        self.s_region *= np.exp(-self.clearance_rate * self.dt)
        self.i_region *= np.exp(-self.clearance_rate * self.dt)

    def clearance_step_thresholded(self):
        self.s_region[self.s_region > 1] *= np.exp(-self.clearance_rate * self.dt)
        self.i_region[self.i_region > 1] *= np.exp(-self.clearance_rate * self.dt)

    def injection(self, seed, amount=1):
        """inject infected agents into seed"""
        self.i_region[seed] = amount

    def trans_step(self):
        infected = self.s_region * (
            1 - np.exp(-self.trans_rate * self.dt * self.i_region)
        )

        self.s_region -= infected
        self.i_region += infected

    def trans_step_thresholded(self):
        infected = self.s_region[self.i_region > 1] * (
                1 - np.exp(-self.trans_rate[self.i_region > 1] * self.dt * self.i_region[self.i_region > 1])
        )

        self.s_region[self.i_region > 1] -= infected
        self.i_region[self.i_region > 1] += infected

    def record_history_region(self):
        self.s_region_history = np.append(
            self.s_region_history,
            self.s_region[np.newaxis, :], axis=0
        )
        self.i_region_history = np.append(
            self.i_region_history,
            self.i_region[np.newaxis, :], axis=0
        )

    def record_history_edge(self):
        self.s_edge_history = np.append(
            self.s_edge_history,
            self.s_edge.sum(axis=0)[np.newaxis, :], axis=0
        )
        self.i_edge_history = np.append(
            self.i_edge_history,
            self.i_edge.sum(axis=0)[np.newaxis, :], axis=0
        )
