from config import RG, INNOVATIVE_FORM

def choice_prob(prob_dict):
    return RG.choice([*prob_dict], p=[*prob_dict.values()])

def compute_prop_communicated_innovative(model):
    #TODO: optimize count?
    n_utterances = len(model.communicated)
    stat = model.communicated.count(INNOVATIVE_FORM)/n_utterances if n_utterances > 0 else 0
    # Empty variable, so only count proportion per iteration is calculated
    model.communicated = []
    return stat