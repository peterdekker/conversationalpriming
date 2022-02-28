from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from collections import defaultdict

from agent import Agent
from config import RG, STEPS_UPDATE_AGENT_COLOR
import util


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height, width, init_prop_innovating_agents, init_prop_innovative_innovating, init_prop_innovative_conservating, boost, surprisal, entropy, repeats):
        '''
        Initialize field
        '''
        assert height % 1 == 0
        assert width % 1 == 0
        assert init_prop_innovating_agents >= 0 and init_prop_innovating_agents <= 1
        assert init_prop_innovative_innovating >= 0 and init_prop_innovative_innovating <= 1
        assert init_prop_innovative_conservating >= 0 and init_prop_innovative_conservating <= 1
        assert boost >= 0 and boost <= 1
        assert type(surprisal) == bool
        assert type(entropy) == bool
        assert type(repeats) == bool


        self.height = height
        self.width = width
        self.init_prop_innovating_agents = init_prop_innovating_agents
        self.boost = boost
        self.surprisal = surprisal
        self.entropy = entropy
        self.repeats = repeats

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0

        # Contains utterances of last step emptied after prop_innovative calculation at end of step
        self.communicated = defaultdict(list)
        self.n_communicated = defaultdict(lambda: [0])
        # Contains proportion innovative of all timesteps
        self.prop_innovative = defaultdict(list)

        ## TODO: maybe later move data initialization from agent to model,
        # so defining these here makes more sense
        self.persons = ["1sg", "2sg", "3sg"]
        self.speaker_types = [False, True]
        ##

        # Averages of last N timesteps of prop_innovative, for whole model
        self.datacollector = DataCollector(
            {
                "prop_innovative_1sg_innovating_avg": util.compute_prop_innovative_1sg_innovating_avg,
                "prop_innovative_2sg_innovating_avg": util.compute_prop_innovative_2sg_innovating_avg,
                "prop_innovative_3sg_innovating_avg": util.compute_prop_innovative_3sg_innovating_avg,
                "prop_innovative_1sg_conservating_avg": util.compute_prop_innovative_1sg_conservating_avg,
                "prop_innovative_2sg_conservating_avg": util.compute_prop_innovative_2sg_conservating_avg,
                "prop_innovative_3sg_conservating_avg": util.compute_prop_innovative_3sg_conservating_avg,
                "prop_innovative_1sg_total_avg": util.compute_prop_innovative_1sg_total_avg,
                "prop_innovative_2sg_total_avg": util.compute_prop_innovative_2sg_total_avg,
                "prop_innovative_3sg_total_avg": util.compute_prop_innovative_3sg_total_avg,
                "prop_innovative_1sg_innovating_internal": util.compute_prop_innovative_1sg_innovating_internal,
                "prop_innovative_2sg_innovating_internal": util.compute_prop_innovative_2sg_innovating_internal,
                "prop_innovative_3sg_innovating_internal": util.compute_prop_innovative_3sg_innovating_internal,
                "prop_innovative_1sg_conservating_internal": util.compute_prop_innovative_1sg_conservating_internal,
                "prop_innovative_2sg_conservating_internal": util.compute_prop_innovative_2sg_conservating_internal,
                "prop_innovative_3sg_conservating_internal": util.compute_prop_innovative_3sg_conservating_internal,
                "prop_innovative_1sg_total_internal": util.compute_prop_innovative_1sg_total_internal,
                "prop_innovative_2sg_total_internal": util.compute_prop_innovative_2sg_total_internal,
                "prop_innovative_3sg_total_internal": util.compute_prop_innovative_3sg_total_internal,
                "prop_innovative_1sg_dominant": util.compute_prop_innovative_1sg_dominant,
                "prop_innovative_2sg_dominant": util.compute_prop_innovative_2sg_dominant,
                "prop_innovative_3sg_dominant": util.compute_prop_innovative_3sg_dominant,
                "n_communicated_1sg": util.compute_n_communicated_1sg_avg,
                "n_communicated_2sg": util.compute_n_communicated_2sg_avg,
                "n_communicated_3sg": util.compute_n_communicated_3sg_avg,
            }
        )


        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for i, cell in enumerate(self.grid.coord_iter()):
            x = cell[1]
            y = cell[2]
            innovating = RG.random() < self.init_prop_innovating_agents
            agent = Agent((x, y), innovating, init_prop_innovative_innovating if innovating else init_prop_innovative_conservating, self)
            self.grid.position_agent(agent, (x, y))
            self.schedule.add(agent)

        self.agents = [a for a, x, y in self.grid.coord_iter()]
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
        if self.steps % STEPS_UPDATE_AGENT_COLOR == 0:
            util.update_prop_innovative_agents(self.agents)

        # Compute model prop
        util.update_prop_innovative_model(self, self.persons, self.speaker_types, self.prop_innovative)
        
        
        self.steps += 1
    

