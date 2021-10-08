import numpy as np
import logging
import sys
from mesa.visualization.UserParam import UserSettableParameter

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
RG = np.random.default_rng()

HEIGHT = 6
WIDTH = 6
PROPORTION_INNOVATIVE = 0.0

model_params = {
    "height": {"ui": HEIGHT, "script": HEIGHT},
    "width": {"ui": WIDTH, "script": WIDTH},
    "proportion_innovative": {"ui": UserSettableParameter("slider", "Proportion L2", PROPORTION_INNOVATIVE, 0.0, 1.0, 0.1), "script": PROPORTION_INNOVATIVE}
}

model_params_ui = {k:v["ui"] for k,v in model_params.items()}
model_params_script = {k:v["script"] for k,v in model_params.items()}

BOOST = 0.01