from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from model import Model

from config import model_params_ui, HEIGHT, WIDTH
from util import compute_colours


def draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    if agent.innovating:
        portrayal = {"Shape": "rect", "w": 0.5, "h": 0.5, "Filled": "true", "Layer": 0}
    else:
        portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = compute_colours(agent)
    portrayal["stroke_color"] = "hsl(0,0%,100%)" # white

    return portrayal


canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)
communicated_1sg_chart = ChartModule([{"Label": "prop_innovative_1sg_innovating_avg", "Color": "Green"}, {"Label": "prop_innovative_1sg_conservating_avg", "Color": "Black"}])
communicated_2sg_chart = ChartModule([{"Label": "prop_innovative_2sg_innovating_avg", "Color": "Green"}, {"Label": "prop_innovative_2sg_conservating_avg", "Color": "Black"}])
communicated_3sg_chart = ChartModule([{"Label": "prop_innovative_3sg_innovating_avg", "Color": "Green"}, {"Label": "prop_innovative_3sg_conservating_avg", "Color": "Black"}])
internal_1sg_chart = ChartModule([{"Label": "prop_innovative_1sg_innovating_internal", "Color": "Green"}, {"Label": "prop_innovative_1sg_conservating_internal", "Color": "Black"}])
internal_2sg_chart = ChartModule([{"Label": "prop_innovative_2sg_innovating_internal", "Color": "Green"}, {"Label": "prop_innovative_2sg_conservating_internal", "Color": "Black"}])
internal_3sg_chart = ChartModule([{"Label": "prop_innovative_3sg_innovating_internal", "Color": "Green"}, {"Label": "prop_innovative_3sg_conservating_internal", "Color": "Black"}])
n_communicated_chart = ChartModule([{"Label": "n_communicated_1sg", "Color": "Red"}, {"Label": "n_communicated_2sg", "Color": "Blue"},{"Label": "n_communicated_3sg", "Color": "Yellow"}])

server = ModularServer(Model,
                       [canvas_element, communicated_1sg_chart, communicated_2sg_chart, communicated_3sg_chart, internal_1sg_chart, internal_2sg_chart, internal_3sg_chart, n_communicated_chart],
                       "Conversational priming", model_params_ui)
