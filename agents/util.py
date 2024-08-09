import numpy as np
import os

from agents.config import RG, INNOVATIVE_FORM
def choice_prob(prob_dict, inverse=False):
    probs = [*prob_dict.values()]
    if inverse:
        probs = [1-p for p in probs]
    return RG.choice([*prob_dict], p=probs)

def mymean(stats_list):
    return np.mean(stats_list) if len(stats_list)>0 else 0


#### DataCollector functions, called every iteration: take mean of last AVG_WINDOW values

# Communicated measures 
def compute_prop_innovative_1sg_conservator_avg(model):
    last_stat = compute_prop_innovative(model.communicated["1sg",False])#[-AVG_WINDOW_STATS:]
    return last_stat # mymean(last_stats)

# def compute_prop_innovative_2sg_conservator_avg(model):
#     last_stat = compute_prop_innovative(model.communicated["2sg",False])#[-AVG_WINDOW_STATS:]
#     return last_stat # mymean(last_stats)

def compute_prop_innovative_3sg_conservator_avg(model):
    last_stat = compute_prop_innovative(model.communicated["3sg",False])#[-AVG_WINDOW_STATS:]
    return last_stat # mymean(last_stats)

def compute_prop_innovative_1sg_innovator_avg(model):
    last_stat = compute_prop_innovative(model.communicated["1sg",True])#[-AVG_WINDOW_STATS:]
    return last_stat # mymean(last_stats)

# def compute_prop_innovative_2sg_innovator_avg(model):
#     last_stat = compute_prop_innovative(model.communicated["2sg",True][)[-AVG_WINDOW_STATS:]
#     return last_stat # mymean(last_stats)

def compute_prop_innovative_3sg_innovator_avg(model):
    last_stat = compute_prop_innovative(model.communicated["3sg",True])#[-AVG_WINDOW_STATS:]
    return last_stat # mymean(last_stats)

def compute_prop_innovative_1sg_total_avg(model):
    last_stat = compute_prop_innovative(model.communicated["1sg",None])#[-AVG_WINDOW_STATS:]
    return last_stat # mymean(last_stats)

# def compute_prop_innovative_2sg_total_avg(model):
#     last_stat = compute_prop_innovative(model.communicated["2sg",None][)[-AVG_WINDOW_STATS:]
#     return last_stat # mymean(last_stats)

def compute_prop_innovative_3sg_total_avg(model):
    last_stat = compute_prop_innovative(model.communicated["3sg",None])#[-AVG_WINDOW_STATS:]
    return last_stat # mymean(last_stats)

## Internal measures

def compute_internal(agents, person, innovator):
    # If innovator==None, compute internal prob for all agents
    probs = [agent.forms[person][INNOVATIVE_FORM] for agent in agents if agent.innovator==innovator or innovator==None]
    return mymean(probs)

def compute_prop_innovative_1sg_conservator_internal(model):
    return compute_internal(model.agents_list, "1sg", False)

# def compute_prop_innovative_2sg_conservator_internal(model):
#     return compute_internal(model.agents_list, "2sg", False)

def compute_prop_innovative_3sg_conservator_internal(model):
    return compute_internal(model.agents_list, "3sg", False)

def compute_prop_innovative_1sg_innovator_internal(model):
    return compute_internal(model.agents_list, "1sg", True)

# def compute_prop_innovative_2sg_innovator_internal(model):
#     return compute_internal(model.agents_list, "2sg", True)

def compute_prop_innovative_3sg_innovator_internal(model):
    return compute_internal(model.agents_list, "3sg", True)

def compute_prop_innovative_1sg_total_internal(model):
    return compute_internal(model.agents_list, "1sg", None)

# def compute_prop_innovative_2sg_total_internal(model):
#     return compute_internal(model.agents_list, "2sg", None)

def compute_prop_innovative_3sg_total_internal(model):
    return compute_internal(model.agents_list, "3sg", None)


def update_prop_innovative_agents(agents):
    for agent in agents:
        agent.prop_innovative = compute_prop_innovative(agent.communicated)
#####

# Method can be applied to communicated_list of agent or model
def compute_prop_innovative(communicated_list):
    #TODO: optimize count?
    n_utterances = len(communicated_list)
    stat = communicated_list.count(INNOVATIVE_FORM)/n_utterances if n_utterances > 0 else 0
    return stat

def update_communicated(form, person, speaker_type, model, agent):
    # For model: store forms per person and agent type
    model.communicated[person,speaker_type].append(form)

    # Also store forms for both speaker_types together
    model.communicated[person,None].append(form)
    
    if model.browser_visualization:
        print("Browser visualization")
        # # For agent: store all forms together, regardless of person and speaker type 
        agent.communicated.append(form)

def compute_colours(agent):
    return "#04b529" if agent.prop_innovative > 0.5 else "#000000"


def colour_str(c):
    return f"hsl({c[0]},{c[1]}%,{c[2]}%)"

def create_output_dir(output_dir):
    # Check if dir exists only for test scripts,
    # in normal cases dir should be created once and not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)