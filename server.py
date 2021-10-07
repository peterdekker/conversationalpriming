from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from model import Model

from config import HEIGHT, WIDTH


def draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = agent.colours
    portrayal["stroke_color"] = "rgb(0,0,0)"

    return portrayal


canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)
communicated_chart = ChartModule([{"Label": "prop_communicated_1", "Color": "Blue"}])

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    #proportion_l2": UserSettableParameter("slider", "Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1),
}

server = ModularServer(Model,
                       [canvas_element, communicated_chart],
                       "Conversational priming", model_params)
