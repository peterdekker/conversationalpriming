from config import RG, INNOVATIVE_FORM

def choice_prob(prob_dict):
    return RG.choice([*prob_dict], p=[*prob_dict.values()])


def compute_prop_innovative_1sg_conservating_speaker(model):
    return compute_prop_innovative(model.communicated["1sg",False])

def compute_prop_innovative_2sg_conservating_speaker(model):
    return compute_prop_innovative(model.communicated["2sg",False])

def compute_prop_innovative_3sg_conservating_speaker(model):
    return compute_prop_innovative(model.communicated["3sg",False])

def compute_prop_innovative_1sg_innovating_speaker(model):
    return compute_prop_innovative(model.communicated["1sg",True])

def compute_prop_innovative_2sg_innovating_speaker(model):
    return compute_prop_innovative(model.communicated["2sg",True])

def compute_prop_innovative_3sg_innovating_speaker(model):
    return compute_prop_innovative(model.communicated["3sg",True])

# Obj can be: model or agent
def compute_prop_innovative(communicated_list):
    #TODO: optimize count?
    n_utterances = len(communicated_list)
    stat = communicated_list.count(INNOVATIVE_FORM)/n_utterances if n_utterances > 0 else 0
    # Empty variable, so only count proportion per iteration is calculated
    communicated_list.clear()
    return stat

def update_stats(form, person, agent_type, model, agent):
    # For model: store forms per person and agent type
    model.communicated[person,agent_type].append(form)

    # For agent: store all forms together, regardless of person and agent type 
    agent.communicated.append(form)

def compute_colours(agent):
    l = agent.prop_communicated_innovative * 50
    # HSL: H->0-360,  S->0-100%, L->100% L50% is maximum color, 100% is white
    return colour_str([110, 90, l])


def colour_str(c):
    return f"hsl({c[0]},{c[1]}%,{c[2]}%)"