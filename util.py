from config import RG

def choice_prob(prob_dict):
    return RG.choice(prob_dict.keys(), p=prob_dict.values())