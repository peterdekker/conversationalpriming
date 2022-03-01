import numpy as np
import os

from config import RG, INNOVATIVE_FORM, AVG_WINDOW_STATS

def choice_prob(prob_dict):
    return RG.choice([*prob_dict], p=[*prob_dict.values()])

def mymean(stats_list):
    return np.mean(stats_list) if len(stats_list)>0 else 0


#### DataCollector functions, called every iteration: take mean of last AVG_WINDOW values
# TODO: call these methods _model_?

# Communicated measures 
def compute_prop_innovative_1sg_conservating_avg(model):
    last_stats = model.prop_innovative["1sg",False][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_2sg_conservating_avg(model):
    last_stats = model.prop_innovative["2sg",False][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_3sg_conservating_avg(model):
    last_stats = model.prop_innovative["3sg",False][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_1sg_innovating_avg(model):
    last_stats = model.prop_innovative["1sg",True][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_2sg_innovating_avg(model):
    last_stats = model.prop_innovative["2sg",True][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_3sg_innovating_avg(model):
    last_stats = model.prop_innovative["3sg",True][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_1sg_total_avg(model):
    last_stats = model.prop_innovative["1sg",None][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_2sg_total_avg(model):
    last_stats = model.prop_innovative["2sg",None][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_prop_innovative_3sg_total_avg(model):
    last_stats = model.prop_innovative["3sg",None][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

## Internal measures

def compute_internal(agents, person, innovating):
    # If innovating==None, compute internal prob for all agents
    probs = [agent.forms[person][INNOVATIVE_FORM] for agent in agents if agent.innovating==innovating or innovating==None]
    return mymean(probs)

def compute_dominant(agents, person, innovating):
    if innovating == None:
        speakers_type = agents
    else:
        speakers_type = [agent for agent in agents if agent.innovating==innovating]
    speakers_dominant = [agent for agent in speakers_type if agent.forms[person][INNOVATIVE_FORM]>= 0.5]
    prop_speakers_dominant = len(speakers_dominant)/len(speakers_type)
    return prop_speakers_dominant

def compute_prop_innovative_1sg_conservating_internal(model):
    return compute_internal(model.agents, "1sg", False)

def compute_prop_innovative_2sg_conservating_internal(model):
    return compute_internal(model.agents, "2sg", False)

def compute_prop_innovative_3sg_conservating_internal(model):
    return compute_internal(model.agents, "3sg", False)

def compute_prop_innovative_1sg_innovating_internal(model):
    return compute_internal(model.agents, "1sg", True)

def compute_prop_innovative_2sg_innovating_internal(model):
    return compute_internal(model.agents, "2sg", True)

def compute_prop_innovative_3sg_innovating_internal(model):
    return compute_internal(model.agents, "3sg", True)

def compute_prop_innovative_1sg_total_internal(model):
    return compute_internal(model.agents, "1sg", None)

def compute_prop_innovative_2sg_total_internal(model):
    return compute_internal(model.agents, "2sg", None)

def compute_prop_innovative_3sg_total_internal(model):
    return compute_internal(model.agents, "3sg", None)

# Dominant measures

def compute_prop_1sg_conservating_dominant(model):
    return compute_dominant(model.agents, "1sg", False)

def compute_prop_2sg_conservating_dominant(model):
    return compute_dominant(model.agents, "2sg", False)

def compute_prop_3sg_conservating_dominant(model):
    return compute_dominant(model.agents, "3sg", False)

def compute_prop_1sg_innovating_dominant(model):
    return compute_dominant(model.agents, "1sg", True)

def compute_prop_2sg_innovating_dominant(model):
    return compute_dominant(model.agents, "2sg", True)

def compute_prop_3sg_innovating_dominant(model):
    return compute_dominant(model.agents, "3sg", True)

def compute_prop_1sg_total_dominant(model):
    return compute_dominant(model.agents, "1sg", None)

def compute_prop_2sg_total_dominant(model):
    return compute_dominant(model.agents, "2sg", None)

def compute_prop_3sg_total_dominant(model):
    return compute_dominant(model.agents, "3sg", None)

##
def compute_n_communicated_1sg_avg(model):
    last_stats = model.n_communicated["1sg"][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_n_communicated_2sg_avg(model):
    last_stats = model.n_communicated["2sg"][:-AVG_WINDOW_STATS]
    return mymean(last_stats)

def compute_n_communicated_3sg_avg(model):
    last_stats = model.n_communicated["3sg"][:-AVG_WINDOW_STATS]
    return mymean(last_stats)
######

##### Called every iteration: compute proportion innovative from list of utterances in this iteration.
# List of utterances is cleared after
def update_prop_innovative_model(model, persons, speaker_types, prop_innovative_obj):
    # print(sorted([(k,len(v)) for k,v in model.communicated.items()]))
    for person in persons:
        for speaker_type in speaker_types:
            # Stat per speaker
            stat = compute_prop_innovative(model.communicated[person, speaker_type])
            prop_innovative_obj[person, speaker_type].append(stat)
        # Total stat
        stat_total = compute_prop_innovative(model.communicated[person, None])
        prop_innovative_obj[person, None].append(stat_total)
        # TODO: should clear of communicated_list also happen here?
        model.n_communicated[person].append(0)

def update_prop_innovative_agents(agents):
    for agent in agents:
        agent.prop_innovative = compute_prop_innovative(agent.communicated)
#####

# Method can be applied to communicated_list of agent or model
def compute_prop_innovative(communicated_list):
    #TODO: optimize count?
    n_utterances = len(communicated_list)
    stat = communicated_list.count(INNOVATIVE_FORM)/n_utterances if n_utterances > 0 else 0
    # Empty variable, so only count proportion per iteration is calculated
    communicated_list.clear()
    return stat

def update_communicated(form, person, speaker_type, model, agent):
    # For model: store forms per person and agent type
    model.communicated[person,speaker_type].append(form)
    model.n_communicated[person][-1] +=1

    # Also store forms for both speaker_types together
    model.communicated[person,None].append(form)

    # For agent: store all forms together, regardless of person and speaker type 
    agent.communicated.append(form)

def compute_colours(agent):
    l = agent.prop_innovative * 50
    # HSL: H->0-360,  S->0-100%, L->100% L50% is maximum color, 100% is white
    return colour_str([110, 90, l])


def colour_str(c):
    return f"hsl({c[0]},{c[1]}%,{c[2]}%)"

def create_output_dir(output_dir):
    # Check if dir exists only for test scripts,
    # in normal cases dir should be created once and not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)