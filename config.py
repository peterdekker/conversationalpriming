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
PROPORTION_INNOVATING = 0.2
BOOST = 0.01
SURPRISAL = False
ENTROPY = False
REPEATS = True

model_params = {
    "height": {"ui": HEIGHT, "script": HEIGHT},
    "width": {"ui": WIDTH, "script": WIDTH},
    "proportion_innovating": {"ui": UserSettableParameter("slider", "Proportion innovating", PROPORTION_INNOVATING, 0.0, 1.0, 0.1), "script": PROPORTION_INNOVATING},
    "boost": {"ui": UserSettableParameter("slider", "Boost", BOOST, 0.0, 1.0, 0.01), "script": BOOST},
    "surprisal": {"ui": UserSettableParameter('checkbox', 'Surprisal', value=SURPRISAL), "script": SURPRISAL},
    "entropy": {"ui": UserSettableParameter('checkbox', 'Entropy', value=ENTROPY), "script": ENTROPY},
    "repeats": {"ui": UserSettableParameter('checkbox', 'Repeats', value=REPEATS), "script": REPEATS}
}

model_params_ui = {k:v["ui"] for k,v in model_params.items()}
model_params_script = {k:v["script"] for k,v in model_params.items()}
