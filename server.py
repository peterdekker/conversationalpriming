from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, NetworkModule
#from mesa.visualization.UserParam import UserSettableParameter

from agents.model import Model

from agents.config import model_params_ui
from agents.util import compute_colours


def draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    if agent.innovator:
        portrayal = {"Shape": "rect", "w": 0.5, "h": 0.5, "Filled": "true", "Layer": 0}
    else:
        portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = compute_colours(agent)
    portrayal["stroke_color"] = "hsl(0,0%,100%)" # white

    return portrayal

def network_portrayal(G):
    # The model ensures there is 0 or 1 agent per node

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "id": node_id,
            "size": 3 if agentsInNode else 1,
            "color": compute_colours(agentsInNode[0]), #"#007959", # "#CC0000" if not agents or agents[0].wealth == 0 else "#007959",
            "label": None
            if not agentsInNode
            else f"Agent:{agentsInNode[0].unique_id}",
        }
        for (node_id, agentsInNode) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {"id": edge_id, "source": source, "target": target, "color": "#000000"}
        for edge_id, (source, target) in enumerate(G.edges)
    ]

    return portrayal


canvas_element = NetworkModule(network_portrayal, 500, 500)
## Old random mixing grid
# canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)


communicated_1sg_chart = ChartModule([{"Label": "prop_innovative_1sg_innovator_avg", "Color": "Green"}, {"Label": "prop_innovative_1sg_conservator_avg", "Color": "Black"}, {"Label": "prop_innovative_1sg_total_avg", "Color": "Blue"}])
# communicated_2sg_chart = ChartModule([{"Label": "prop_innovative_2sg_innovator_avg", "Color": "Green"}, {"Label": "prop_innovative_2sg_conservator_avg", "Color": "Black"}, {"Label": "prop_innovative_2sg_total_avg", "Color": "Blue"}])
communicated_3sg_chart = ChartModule([{"Label": "prop_innovative_3sg_innovator_avg", "Color": "Green"}, {"Label": "prop_innovative_3sg_conservator_avg", "Color": "Black"}, {"Label": "prop_innovative_3sg_total_avg", "Color": "Blue"}])

internal_1sg_chart = ChartModule([{"Label": "prop_innovative_1sg_innovator_internal", "Color": "Green"}, {"Label": "prop_innovative_1sg_conservator_internal", "Color": "Black"}, {"Label": "prop_innovative_1sg_total_internal", "Color": "Blue"}])
# internal_2sg_chart = ChartModule([{"Label": "prop_innovative_2sg_innovator_internal", "Color": "Green"}, {"Label": "prop_innovative_2sg_conservator_internal", "Color": "Black"}, {"Label": "prop_innovative_2sg_total_internal", "Color": "Blue"}])
internal_3sg_chart = ChartModule([{"Label": "prop_innovative_3sg_innovator_internal", "Color": "Green"}, {"Label": "prop_innovative_3sg_conservator_internal", "Color": "Black"}, {"Label": "prop_innovative_3sg_total_internal", "Color": "Blue"}])

# dominant_1sg_chart = ChartModule([{"Label": "prop_1sg_innovator_dominant", "Color": "Green"}, {"Label": "prop_1sg_conservator_dominant", "Color": "Black"}, {"Label": "prop_1sg_total_dominant", "Color": "Blue"}])
# # dominant_2sg_chart = ChartModule([{"Label": "prop_2sg_innovator_dominant", "Color": "Green"}, {"Label": "prop_2sg_conservator_dominant", "Color": "Black"}, {"Label": "prop_2sg_total_dominant", "Color": "Blue"}])
# dominant_3sg_chart = ChartModule([{"Label": "prop_3sg_innovator_dominant", "Color": "Green"}, {"Label": "prop_3sg_conservator_dominant", "Color": "Black"}, {"Label": "prop_3sg_total_dominant", "Color": "Blue"}])

#n_communicated_chart = ChartModule([{"Label": "n_communicated_1sg", "Color": "Red"}, {"Label": "n_communicated_2sg", "Color": "Blue"},{"Label": "n_communicated_3sg", "Color": "Yellow"}])

server = ModularServer(Model,
                       [canvas_element, communicated_1sg_chart, communicated_3sg_chart, internal_1sg_chart, internal_3sg_chart], # dominant_1sg_chart, dominant_3sg_chart],
                       "Conversational priming", model_params_ui)
