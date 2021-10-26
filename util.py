from config import RG, INNOVATIVE_FORM

def choice_prob(prob_dict):
    return RG.choice([*prob_dict], p=[*prob_dict.values()])

# Obj can be: model or agent
def compute_prop_communicated_innovative(obj):
    #TODO: optimize count?
    n_utterances = len(obj.communicated)
    stat = obj.communicated.count(INNOVATIVE_FORM)/n_utterances if n_utterances > 0 else 0
    # Empty variable, so only count proportion per iteration is calculated
    obj.communicated = []
    return stat

def update_stats(form, model, agent):
    model.communicated.append(form)
    agent.communicated.append(form)

def compute_colours(agent):
    shade = agent.prop_communicated_innovative * 50
    print(shade)
    # HSL: H->0-360,  S->0-100%, L->100% L50% is maximum color, 100% is white
    return colour_str([250, 80, shade]), colour_str([250, 80, shade])


def colour_str(c):
    return f"hsl({c[0]},{c[1]}%,{c[2]}%)"