import numpy as np
import logging
import sys
from mesa.visualization.UserParam import UserSettableParameter

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
RG = np.random.default_rng()
INNOVATIVE_FORM = "1"
STEPS_UPDATE_AGENT_COLOR = 20
AVG_WINDOW_STATS = 10
SMOOTHING_SURPRISAL = 0.01

HEIGHT = 10
WIDTH = 10
PROP_INNOVATING_AGENTS = 0.2
PROP_INNOVATIVE_INNOVATING = 0.9
PROP_INNOVATIVE_CONSERVATING = 0.0
BOOST = 0.01
SURPRISAL = True
ENTROPY = False
REPEATS = True

model_params = {
    "height": {"ui": HEIGHT, "script": HEIGHT},
    "width": {"ui": WIDTH, "script": WIDTH},
    "prop_innovating_agents": {"ui": UserSettableParameter("slider", "Proportion innovating agents", PROP_INNOVATING_AGENTS, 0.0, 1.0, 0.1), "script": PROP_INNOVATING_AGENTS},
    "prop_innovative_innovating": {"ui": UserSettableParameter("slider", "Proportion innovative forms in innovating agents", PROP_INNOVATIVE_INNOVATING, 0.0, 1.0, 0.1), "script": PROP_INNOVATIVE_INNOVATING},
    "prop_innovative_conservating": {"ui": UserSettableParameter("slider", "Proportion innovative forms in conservating agents", PROP_INNOVATIVE_CONSERVATING, 0.0, 1.0, 0.1), "script": PROP_INNOVATIVE_CONSERVATING},
    "boost": {"ui": UserSettableParameter("slider", "Boost", BOOST, 0.0, 1.0, 0.01), "script": BOOST},
    "surprisal": {"ui": UserSettableParameter('checkbox', 'Surprisal', value=SURPRISAL), "script": SURPRISAL},
    "entropy": {"ui": UserSettableParameter('checkbox', 'Entropy', value=ENTROPY), "script": ENTROPY},
    "repeats": {"ui": UserSettableParameter('checkbox', 'Repeats', value=REPEATS), "script": REPEATS}
}

model_params_ui = {k:v["ui"] for k,v in model_params.items()}
model_params_script = {k:v["script"] for k,v in model_params.items()}
