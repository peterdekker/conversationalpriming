from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from agent import Agent
from config import MAX_RADIUS


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height, width):
        '''
        Initialize field
        '''
        assert height % 1 == 0
        assert width % 1 == 0


        self.height = height
        self.width = width
        self.radius = MAX_RADIUS

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0


        self.datacollector = DataCollector(
            {
            }
        )


        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for i, cell in enumerate(self.grid.coord_iter()):
            x = cell[1]
            y = cell[2]
            agent = Agent((x, y), self)
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
        self.steps += 1
