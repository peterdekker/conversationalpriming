from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from agent import Agent
from config import RG
from util import compute_prop_communicated_innovative


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height, width, proportion_innovative):
        '''
        Initialize field
        '''
        assert height % 1 == 0
        assert width % 1 == 0
        assert proportion_innovative >= 0 and proportion_innovative <= 1


        self.height = height
        self.width = width
        self.proportion_innovative = proportion_innovative

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0

        self.communicated = []


        self.datacollector = DataCollector(
            {
                "prop_communicated_innovative": compute_prop_communicated_innovative
            }
        )


        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for i, cell in enumerate(self.grid.coord_iter()):
            x = cell[1]
            y = cell[2]
            innovative = RG.random() < self.proportion_innovative
            agent = Agent((x, y), innovative, self)
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
                agent.prop_communicated_innovative = compute_prop_communicated_innovative(agent)
        
        self.steps += 1

