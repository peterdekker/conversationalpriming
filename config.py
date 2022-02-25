import numpy as np
import logging
import sys
import datetime
from mesa.visualization.UserParam import UserSettableParameter

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
RG = np.random.default_rng()
INNOVATIVE_FORM = "1"
PERSONS = ["1sg", "2sg", "3sg"]
STEPS_UPDATE_AGENT_COLOR = 50
AVG_WINDOW_STATS = 10
SMOOTHING_SURPRISAL = 0.01 # not needed anymore
LAST_N_STEPS_END_GRAPH = 500

IMG_FORMAT = "png"
OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-")}'

# For evaluation script (not browser visualization)
ITERATIONS = [3]
STEPS = [5000]

HEIGHT = 10
WIDTH = 10
INIT_PROP_INNOVATING_AGENTS = 0.2
INIT_PROP_INNOVATIVE_INNOVATING = 0.9
INIT_PROP_INNOVATIVE_CONSERVATING = 0.0
BOOST = 0.01
SURPRISAL = False
ENTROPY = False
REPEATS = True

model_params = {
    "height": {"ui": HEIGHT, "script": HEIGHT},
    "width": {"ui": WIDTH, "script": WIDTH},
    "init_prop_innovating_agents": {"ui": UserSettableParameter("slider", "Proportion innovating agents", INIT_PROP_INNOVATING_AGENTS, 0.0, 1.0, 0.1), "script": INIT_PROP_INNOVATING_AGENTS},
    "init_prop_innovative_innovating": {"ui": UserSettableParameter("slider", "Proportion innovative forms in innovating agents", INIT_PROP_INNOVATIVE_INNOVATING, 0.0, 1.0, 0.1), "script": INIT_PROP_INNOVATIVE_INNOVATING},
    "init_prop_innovative_conservating": {"ui": UserSettableParameter("slider", "Proportion innovative forms in conservating agents", INIT_PROP_INNOVATIVE_CONSERVATING, 0.0, 1.0, 0.1), "script": INIT_PROP_INNOVATIVE_CONSERVATING},
    "boost": {"ui": UserSettableParameter("slider", "Boost", BOOST, 0.0, 1.0, 0.01), "script": BOOST},
    "surprisal": {"ui": UserSettableParameter('checkbox', 'Surprisal', value=SURPRISAL), "script": SURPRISAL},
    "entropy": {"ui": UserSettableParameter('checkbox', 'Entropy', value=ENTROPY), "script": ENTROPY},
    "repeats": {"ui": UserSettableParameter('checkbox', 'Repeats', value=REPEATS), "script": REPEATS}
}

model_params_ui = {k:v["ui"] for k,v in model_params.items()}
model_params_script = {k:v["script"] for k,v in model_params.items()}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "runlabel": ""
}

bool_params = ["surprisal", "entropy", "repeats"]

string_params = ["runlabel"]