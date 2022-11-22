import numpy as np
import logging
import sys
import datetime
from mesa.visualization.UserParam import UserSettableParameter, Slider, Checkbox

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
RG = np.random.default_rng()
INNOVATIVE_FORM = "1"
PERSONS = ["1sg", "2sg", "3sg"]
STEPS_UPDATE_AGENT_COLOR = 50
AVG_WINDOW_STATS = 1
SMOOTHING_SURPRISAL = 0.01 # not needed anymore
LAST_N_STEPS_END_GRAPH = 500

IMG_FORMATS = ["png", "pdf"]
OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-").replace(":",".")}'

# For evaluation script (not browser visualization)
ITERATIONS = [20]
STEPS = [5000]

N_AGENTS = 100
PROP_INNOVATING_AGENTS = 0.2
INIT_PROP_INNOVATIVE_INNOVATING = 0.9
INIT_PROP_INNOVATIVE_CONSERVATING = 0.0
FREQ_3SG = 0.6

BOOST_CONSERVATIVE = 0.01
BOOST_INNOVATIVE = 0.01
FORGET_WEIGHT = 0.00

SURPRISAL = False
ENTROPY = False
REPEATS = True
PRIMING = True
FRIEND_NETWORK = False
INNOVATING_NO_PRIMING = False
INNOVATING_ONLY_BOOST_PRODUCTION = False
N_INTERACTIONS_INTERLOCUTOR = 1

model_params = {
    "n_agents": {"ui": Slider("# agents", N_AGENTS, 0, 1000, 10), "script": N_AGENTS},
    "prop_innovating_agents": {"ui": Slider("Proportion innovating agents", PROP_INNOVATING_AGENTS, 0.0, 1.0, 0.1), "script": PROP_INNOVATING_AGENTS},
    "init_prop_innovative_innovating": {"ui": Slider("Proportion innovative forms in innovating agents", INIT_PROP_INNOVATIVE_INNOVATING, 0.0, 1.0, 0.1), "script": INIT_PROP_INNOVATIVE_INNOVATING},
    "init_prop_innovative_conservating": {"ui": Slider("Proportion innovative forms in conservating agents", INIT_PROP_INNOVATIVE_CONSERVATING, 0.0, 1.0, 0.1), "script": INIT_PROP_INNOVATIVE_CONSERVATING},
    "freq_3sg": {"ui": Slider("Frequency 3sg", FREQ_3SG, 0.0, 1.0, 0.01), "script": FREQ_3SG},
    "boost_conservative": {"ui": Slider("Boost conservative", BOOST_CONSERVATIVE, 0.0, 1.0, 0.01), "script": BOOST_CONSERVATIVE},
    "boost_innovative": {"ui": Slider("Boost innovative", BOOST_INNOVATIVE, 0.0, 1.0, 0.01), "script": BOOST_INNOVATIVE},
    "forget_weight": {"ui": Slider("Forget weight", FORGET_WEIGHT, 0.0, 1.0, 0.01), "script": FORGET_WEIGHT},
    "surprisal": {"ui": Checkbox('Surprisal', value=SURPRISAL), "script": SURPRISAL},
    "entropy": {"ui": Checkbox('Entropy', value=ENTROPY), "script": ENTROPY},
    "repeats": {"ui": Checkbox('Repeats', value=REPEATS), "script": REPEATS},
    "priming": {"ui": Checkbox('priming', value=PRIMING), "script": PRIMING},
    "friend_network": {"ui": Checkbox('Friend network', value=FRIEND_NETWORK), "script": FRIEND_NETWORK},
    "innovating_no_priming": {"ui": Checkbox('Innovating no priming', value=INNOVATING_NO_PRIMING), "script": INNOVATING_NO_PRIMING},
    "innovating_only_boost_production": {"ui": Checkbox('Innovating only boost production', value=INNOVATING_ONLY_BOOST_PRODUCTION), "script": INNOVATING_ONLY_BOOST_PRODUCTION},
    "n_interactions_interlocutor": {"ui": Slider("# interactions per interlocutor", N_INTERACTIONS_INTERLOCUTOR, 1, 100, 1), "script": N_INTERACTIONS_INTERLOCUTOR},
    "browser_visualization": {"ui": True, "script": False},
    "use_grid": {"ui": False, "script": False},
}

model_params_ui = {k:v["ui"] for k,v in model_params.items()}
model_params_script = {k:v["script"] for k,v in model_params.items()}


evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "runlabel": "",
    "plot_from_raw": "",
    "contrast_persons": False
}

bool_params = ["surprisal", "entropy", "repeats", "priming", "friend_network", "innovating_no_priming", "innovating_only_boost_production", "browser_visualization", "use_grid",  "contrast_persons"]

string_params = ["runlabel", "plot_from_raw"]