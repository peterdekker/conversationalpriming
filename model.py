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

    def __init__(self, height, width, proportion_innovating, repeats):
        '''
        Initialize field
        '''
        assert height % 1 == 0
        assert width % 1 == 0
        assert proportion_innovating >= 0 and proportion_innovating <= 1


        self.height = height
        self.width = width
        self.proportion_innovating = proportion_innovating
        self.repeats = repeats

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0

        # Contains utterances of last step emptied after prop_innovative calculation at end of step
        self.communicated = defaultdict(list)
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
            }
        )


        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for i, cell in enumerate(self.grid.coord_iter()):
            x = cell[1]
            y = cell[2]
            innovating = RG.random() < self.proportion_innovating
            agent = Agent((x, y), innovating, self)
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

