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
LAST_N_STEPS_END_GRAPH = 500

IMG_FORMATS = ["png", "pdf"]
OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-").replace(":",".")}'

# For evaluation script (not browser visualization)
ITERATIONS = [50]
STEPS = [5000]

N_AGENTS = 100
PROP_INNOVATOR_AGENTS = 0.2
INIT_PROP_INNOVATIVE_INNOVATOR = 0.9
INIT_PROP_INNOVATIVE_CONSERVATOR = 0.00
BOOST_CONSERVATIVE = 0.01
BOOST_INNOVATIVE =  0.01 # 0.02 #
FORGET_WEIGHT = 0.00
FREQ_3SG =  1/3 #0.5 #

SURPRISAL = False
ENTROPY = False
REPEATS = True
CONVERSATIONAL_PRIMING_PROB = 1.0
FRIEND_NETWORK = False
INNOVATOR_NO_CONVERSATIONAL_PRIMING = False
INNOVATOR_ONLY_BOOST_PRODUCTION = False
N_INTERACTIONS_INTERLOCUTOR = 1

model_params = {
    "n_agents": {"ui": Slider("# agents", N_AGENTS, 0, 1000, 10), "script": N_AGENTS},
    "prop_innovator_agents": {"ui": Slider("Proportion innovator agents", PROP_INNOVATOR_AGENTS, 0.0, 1.0, 0.1), "script": PROP_INNOVATOR_AGENTS},
    "init_prop_innovative_innovator": {"ui": Slider("Proportion innovative forms in innovator agents", INIT_PROP_INNOVATIVE_INNOVATOR, 0.0, 1.0, 0.01), "script": INIT_PROP_INNOVATIVE_INNOVATOR},
    "init_prop_innovative_conservator": {"ui": Slider("Proportion innovative forms in conservator agents", INIT_PROP_INNOVATIVE_CONSERVATOR, 0.0, 1.0, 0.01), "script": INIT_PROP_INNOVATIVE_CONSERVATOR},
    "freq_3sg": {"ui": Slider("Frequency 3sg", FREQ_3SG, 0.0, 1.0, 0.01), "script": FREQ_3SG},
    "boost_conservative": {"ui": Slider("Boost conservative", BOOST_CONSERVATIVE, 0.0, 1.0, 0.01), "script": BOOST_CONSERVATIVE},
    "boost_innovative": {"ui": Slider("Boost innovative", BOOST_INNOVATIVE, 0.0, 1.0, 0.01), "script": BOOST_INNOVATIVE},
    "forget_weight": {"ui": Slider("Forget weight", FORGET_WEIGHT, 0.0, 1.0, 0.01), "script": FORGET_WEIGHT},
    "surprisal": {"ui": Checkbox('Surprisal', value=SURPRISAL), "script": SURPRISAL},
    "entropy": {"ui": Checkbox('Entropy', value=ENTROPY), "script": ENTROPY},
    "repeats": {"ui": Checkbox('Repeats', value=REPEATS), "script": REPEATS},
    "conversational_priming_prob": {"ui": Slider('Conversational priming prob', CONVERSATIONAL_PRIMING_PROB, 0.0, 1.0, 0.1), "script": CONVERSATIONAL_PRIMING_PROB},
    "friend_network": {"ui": Checkbox('Friend network', value=FRIEND_NETWORK), "script": FRIEND_NETWORK},
    "innovator_no_conversational_priming": {"ui": Checkbox('Innovator no conversational_priming', value=INNOVATOR_NO_CONVERSATIONAL_PRIMING), "script": INNOVATOR_NO_CONVERSATIONAL_PRIMING},
    "innovator_only_boost_production": {"ui": Checkbox('Innovator only boost production', value=INNOVATOR_ONLY_BOOST_PRODUCTION), "script": INNOVATOR_ONLY_BOOST_PRODUCTION},
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

bool_params = ["surprisal", "entropy", "repeats", "conversational_priming", "friend_network", "innovator_no_conversational_priming", "innovator_only_boost_production", "browser_visualization", "use_grid",  "contrast_persons"]

string_params = ["runlabel", "plot_from_raw"]