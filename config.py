import numpy as np
import logging
import sys
from mesa.visualization.UserParam import UserSettableParameter

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
RG = np.random.default_rng()
INNOVATIVE_FORM = "2"

HEIGHT = 6
WIDTH = 6
PROPORTION_INNOVATING = 0.2

model_params = {
    "height": {"ui": HEIGHT, "script": HEIGHT},
    "width": {"ui": WIDTH, "script": WIDTH},
    "proportion_innovating": {"ui": UserSettableParameter("slider", "Proportion innovating", PROPORTION_INNOVATING, 0.0, 1.0, 0.1), "script": PROPORTION_INNOVATING}
}

model_params_ui = {k:v["ui"] for k,v in model_params.items()}
model_params_script = {k:v["script"] for k,v in model_params.items()}

BOOST = 0.01