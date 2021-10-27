from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from collections import defaultdict

from agent import Agent
from config import RG
import util


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height, width, proportion_innovating):
        '''
        Initialize field
        '''
        assert height % 1 == 0
        assert width % 1 == 0
        assert proportion_innovating >= 0 and proportion_innovating <= 1


        self.height = height
        self.width = width
        self.proportion_innovating = proportion_innovating

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0

        self.communicated = defaultdict(list)

        self.datacollector = DataCollector(
            {
                "prop_innovative_1sg_innovating_speaker": util.compute_prop_innovative_1sg_innovating_speaker,
                "prop_innovative_2sg_innovating_speaker": util.compute_prop_innovative_2sg_innovating_speaker,
                "prop_innovative_3sg_innovating_speaker": util.compute_prop_innovative_3sg_innovating_speaker,
                "prop_innovative_1sg_conservating_speaker": util.compute_prop_innovative_1sg_conservating_speaker,
                "prop_innovative_2sg_conservating_speaker": util.compute_prop_innovative_2sg_conservating_speaker,
                "prop_innovative_3sg_conservating_speaker": util.compute_prop_innovative_3sg_conservating_speaker,
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

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        '''
        Run one step of the model.
        '''
        
        self.schedule.step()
        self.datacollector.collect(self)

        # Compute agent prop communicated every n steps
        # This also empties variable
        if self.steps % 10 == 0:
            for agent in self.agents:
                agent.prop_communicated_innovative = util.compute_prop_innovative(agent.communicated)
        
        
        self.steps += 1

