from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid, NetworkGrid
from mesa.datacollection import DataCollector

import networkx as nx

from collections import defaultdict

from agents.agent import Agent
from agents.config import RG, STEPS_UPDATE_AGENT_COLOR
from agents.network import create_innovative_agents, create_network_friend_of_friend_fixed_degree, create_network_complete
import agents.util as util

import numpy as np

class Model(Model):
    '''
    Model class
    '''

    def __init__(self, n_agents, prop_innovator_agents, init_prop_innovative_innovator, init_prop_innovative_conservator, freq_3sg, boost_conservative, boost_innovative, forget_weight, surprisal, entropy, repeats, conversational_priming, friend_network, innovator_no_conversational_priming, innovator_only_boost_production, n_interactions_interlocutor, browser_visualization, use_grid):
        '''
        Initialize field
        '''
        assert n_agents % 1 == 0
        assert prop_innovator_agents >= 0 and prop_innovator_agents <= 1
        assert init_prop_innovative_innovator >= 0 and init_prop_innovative_innovator <= 1
        assert init_prop_innovative_conservator >= 0 and init_prop_innovative_conservator <= 1
        assert freq_3sg >= 0 and freq_3sg <= 1

        assert boost_conservative >= 0 and boost_conservative <= 1
        assert boost_innovative >= 0 and boost_innovative <= 1
        assert forget_weight >= 0 and forget_weight <= 1
        assert type(surprisal) == bool
        assert type(entropy) == bool
        assert type(repeats) == bool
        assert type(conversational_priming) == bool
        assert type(friend_network) == bool
        assert type(innovator_no_conversational_priming) == bool
        assert type(innovator_only_boost_production) == bool
        assert n_interactions_interlocutor >= 1 and n_interactions_interlocutor <= 100
        assert type(browser_visualization) == bool
        assert type(use_grid)==bool

        if (surprisal or entropy) and (init_prop_innovative_conservator == 0.0 or init_prop_innovative_innovator == 0.0):
            raise ValueError(
                "If surprisal or entropy is on, the proportion of innovative forms in innovator and conservator agents have to be > 0.0; to prevent NaN values in surprisal calculations.")

        self.n_agents = int(n_agents)
        self.prop_innovator_agents = prop_innovator_agents
        self.boost_conservative = boost_conservative
        self.boost_innovative = boost_innovative
        self.forget_weight = forget_weight
        self.surprisal = surprisal
        self.entropy = entropy
        self.repeats = repeats
        self.conversational_priming = conversational_priming
        self.friend_network = friend_network
        self.innovator_no_conversational_priming = innovator_no_conversational_priming
        self.innovator_only_boost_production = innovator_only_boost_production
        self.n_interactions_interlocutor = int(n_interactions_interlocutor)
        self.browser_visualization = browser_visualization
        self.use_grid = use_grid
        self.freq_3sg = freq_3sg

        self.schedule = RandomActivation(self)
        self.steps = 0

        # Diagnostic variables
        self.n_total_boosts = 0
        self.n_total_forgets = 0

        # Contains utterances of last step emptied after prop_innovative calculation at end of step
        self.communicated = defaultdict(list)
        # self.n_communicated = defaultdict(lambda: [0])
        # Contains proportion innovative of all timesteps
        self.prop_innovative = defaultdict(list)

        # TODO: maybe later move data initialization from agent to model,
        # so defining these here makes more sense
        self.persons = ["1sg", "2sg", "3sg"]
        self.speaker_types = [False, True]
        ##

        # Averages of last N timesteps of prop_innovative, for whole model
        self.datacollector = DataCollector(
            {
                "prop_innovative_1sg_innovator_avg": util.compute_prop_innovative_1sg_innovator_avg,
                #"prop_innovative_2sg_innovator_avg": util.compute_prop_innovative_2sg_innovator_avg,
                "prop_innovative_3sg_innovator_avg": util.compute_prop_innovative_3sg_innovator_avg,
                "prop_innovative_1sg_conservator_avg": util.compute_prop_innovative_1sg_conservator_avg,
                #"prop_innovative_2sg_conservator_avg": util.compute_prop_innovative_2sg_conservator_avg,
                "prop_innovative_3sg_conservator_avg": util.compute_prop_innovative_3sg_conservator_avg,
                "prop_innovative_1sg_total_avg": util.compute_prop_innovative_1sg_total_avg,
                #"prop_innovative_2sg_total_avg": util.compute_prop_innovative_2sg_total_avg,
                "prop_innovative_3sg_total_avg": util.compute_prop_innovative_3sg_total_avg,
                "prop_innovative_1sg_innovator_internal": util.compute_prop_innovative_1sg_innovator_internal,
                #"prop_innovative_2sg_innovator_internal": util.compute_prop_innovative_2sg_innovator_internal,
                "prop_innovative_3sg_innovator_internal": util.compute_prop_innovative_3sg_innovator_internal,
                "prop_innovative_1sg_conservator_internal": util.compute_prop_innovative_1sg_conservator_internal,
                #"prop_innovative_2sg_conservator_internal": util.compute_prop_innovative_2sg_conservator_internal,
                "prop_innovative_3sg_conservator_internal": util.compute_prop_innovative_3sg_conservator_internal,
                "prop_innovative_1sg_total_internal": util.compute_prop_innovative_1sg_total_internal,
                #"prop_innovative_2sg_total_internal": util.compute_prop_innovative_2sg_total_internal,
                "prop_innovative_3sg_total_internal": util.compute_prop_innovative_3sg_total_internal,
                # "prop_1sg_innovator_dominant": util.compute_prop_1sg_innovator_dominant,
                # "prop_2sg_innovator_dominant": util.compute_prop_2sg_innovator_dominant,
                # "prop_3sg_innovator_dominant": util.compute_prop_3sg_innovator_dominant,
                # "prop_1sg_conservator_dominant": util.compute_prop_1sg_conservator_dominant,
                # "prop_2sg_conservator_dominant": util.compute_prop_2sg_conservator_dominant,
                # "prop_3sg_conservator_dominant": util.compute_prop_3sg_conservator_dominant,
                # "prop_1sg_total_dominant": util.compute_prop_1sg_total_dominant,
                # "prop_2sg_total_dominant": util.compute_prop_2sg_total_dominant,
                # "prop_3sg_total_dominant": util.compute_prop_3sg_total_dominant,
                #"n_communicated_1sg": util.compute_n_communicated_1sg_avg,
                #"n_communicated_2sg": util.compute_n_communicated_2sg_avg,
                #"n_communicated_3sg": util.compute_n_communicated_3sg_avg,
            }
        )

        if not self.use_grid: # use network, default
            agent_types, agents = create_innovative_agents(
                    self.n_agents, self.prop_innovator_agents)
            if self.friend_network:
                self.G = create_network_friend_of_friend_fixed_degree(stranger_connect_prob=0.3, conservator_friend_of_friend_connect_prob=1.0, innovator_friend_of_friend_connect_prob=0.3, max_degree=10, agent_types=agent_types, agents=agents)
            else:
                self.G = create_network_complete(self.n_agents, agent_types)
            self.grid = NetworkGrid(self.G)
            for node_name, node_data in self.G.nodes(data=True):
                innovator = node_data["innovator"]
                agent = Agent(
                    node_name, innovator, init_prop_innovative_innovator if innovator else init_prop_innovative_conservator, self)
                # Add the agent to the graph node
                self.grid.place_agent(agent, node_name)
                self.schedule.add(agent)
        ## Old random mixing grid code
        if self.use_grid:
            # Set up agents
            # We use a grid iterator that returns
            # the coordinates of a cell as well as
            # its contents. (coord_iter)
            self.grid = SingleGrid(10,10, torus=True)
            for i, cell in enumerate(self.grid.coord_iter()):
                x = cell[1]
                y = cell[2]
                innovator = RG.random() < self.prop_innovator_agents
                agent = Agent(
                    (x, y), innovator, init_prop_innovative_innovator if innovator else init_prop_innovative_conservator, self)
                self.grid.position_agent(agent, (x, y))
                self.schedule.add(agent)

        self.agents = self.schedule.agents
        if self.browser_visualization:
            util.update_prop_innovative_agents(self.agents)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        '''
        Run one step of the model.
        '''

        self.schedule.step()
        self.datacollector.collect(self)

        # Compute agent prop, every N iterations
        # This also empties variable
        # TODO: Remove this when running as script
        if self.browser_visualization and self.steps % STEPS_UPDATE_AGENT_COLOR == 0:
            util.update_prop_innovative_agents(self.agents)

        # Compute model prop
        util.update_prop_innovative_model(
            self, self.persons, self.speaker_types, self.prop_innovative)

        self.steps += 1
        # if self.steps==1000:
        #     print(f"forgets: {self.n_total_forgets}. boosts: {self.n_total_boosts}. forgets/boosts: {self.n_total_forgets/self.n_total_boosts}")
