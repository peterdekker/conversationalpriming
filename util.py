from config import RG

def choice_prob(prob_dict):
    return RG.choice([*prob_dict], p=[*prob_dict.values()])