import mesa

from verb import Verb
from util import choice_prob, update_communicated
from config import RG, logging, SMOOTHING_SURPRISAL, PERSONS, INNOVATIVE_FORM
import numpy as np


class Agent(mesa.Agent):
    def __init__(self, pos, innovator, prop_innovative_forms, model):


        super().__init__(pos, model)

        self.pos = pos
        self.innovator = innovator
        # TODO: later possibly add corpus probabilities
        # TODO: Move initialization outside agent?
        #self.verb_concepts = ["a"]
        self.persons = PERSONS
        self.forms = {}
        #for c in self.verb_concepts:
        for p in self.persons:
            self.forms[p] = {"0":1-prop_innovative_forms, "1": prop_innovative_forms}

        self.question_answer_mapping = {"1sg":"2sg", "2sg":"1sg", "3sg":"3sg"}

        # Variables for stats (for colours)
        self.communicated = []
        self.prop_innovative = []

        freq_1sg_2sg = (1-self.model.freq_3sg)/2
        self.person_weights = [freq_1sg_2sg, freq_1sg_2sg, self.model.freq_3sg]


    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        if not self.model.use_grid: # default> use network
            neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
            neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
            listener = RG.choice(neighbors)
        
        if self.model.use_grid:
             listener = RG.choice([a for a in self.model.agents if a.pos != self.pos])
        for i in range(self.model.n_interactions_interlocutor):
            question = self.create_question()
            answer = listener.receive_question_reply(question)
            if answer is not None:
                self.receive_answer(answer)
            
            # After every interaction, forget random token
            # Forget one form for every person. This makes frequency assymetry between persons relevant.
            if self.model.forget_weight > 0.0:
                for person in self.persons:
                    # Simulate sampling from list, by sampling with probability of form
                    # So if 0.9 probability for token 1: 0.9 chance to forget 1
                    sample_token = min(self.forms[person], key=self.forms[person].get) # choice_prob(self.forms[person], inverse=True) # 
                    self.forget_form(person, sample_token)


    def create_question(self):
        self.person_question = RG.choice(self.persons, p=self.person_weights)
        form_question = choice_prob(self.forms[self.person_question])
        self.boost_form(self.person_question, form_question)
        # Add to stats
        update_communicated(form_question, self.person_question, self.innovator, self.model, self)
        

        signal_question = Verb(person=self.person_question, form=form_question)
        return signal_question

    # Methods used when agent listens

    def receive_question_reply(self, signal):
        person_question, form_question = signal.get_content()
        if not (self.model.innovator_only_boost_production and self.innovator):
            self.boost_form(person_question, form_question)

        if self.model.repeats:
            person_answer = self.question_answer_mapping[person_question]
            if person_answer == person_question and self.model.conv_priming and not (self.model.innovator_no_conv_priming and self.innovator):
                # 3sg: instead of using own forms library, just repeat form from question
                form_answer = form_question
            else:
                # Other cases: use form from own library
                form_answer = choice_prob(self.forms[person_answer])
            self.boost_form(person_answer, form_answer)
            # Add to stats
            update_communicated(form_answer, person_answer, self.innovator, self.model, self)

            signal_answer = Verb(person=person_answer, form=form_answer)
        else:
            # In no-repeats model, answer with yes/no, so no form used
            signal_answer=None
        return signal_answer
    
    def receive_answer(self, signal):
        person_answer, form_answer = signal.get_content()
        if not (self.model.innovator_only_boost_production and self.innovator):
            self.boost_form(person_answer, form_answer)

        # Reset internal person question variable
        self.person_question = None
    
    def boost_form(self, person, form):
        # if (self.innovator and form != INNOVATIVE_FORM):
        #     return
        SURPRISAL_THRESHOLD = 1000000
        prob_dict = self.forms[person]
        if form == INNOVATIVE_FORM:
            boost = self.model.boost_innovative
        else:
            boost = self.model.boost_conservative
        if self.model.surprisal:
            #surprisal = min(-np.log2(prob_dict[form]), SURPRISAL_THRESHOLD)
            surprisal = -np.log2(prob_dict[form])
            boost = boost * surprisal
        if self.model.entropy and not self.model.surprisal:
            px = prob_dict[form]
            # surprisal = min(-np.log2(px), SURPRISAL_THRESHOLD)
            surprisal = -np.log2(px)
            entropy = px * surprisal
            boost = boost * 2 * entropy
        new_total = 1.0 + boost
        # Add BOOST to this form and scale by new total, scale other forms by new total
        self.forms[person] = {f: (prob+boost)/new_total if f==form else prob/new_total for f, prob in prob_dict.items()}

        # Counter for diagnostic purposes
        self.model.n_total_boosts += 1
    
    def forget_form(self, person, form):
        prob_dict = self.forms[person]
        boost = self.model.forget_weight
        new_total = 1.0 + boost
        # Forget this form by boosting other form
        other_form = "1" if form=="0" else "0"
        # print(f"Perform forget for {form} by boosting {other_form}")
        # print(self.forms[person])
        # Add BOOST to this form and scale by new total, scale other forms by new total
        self.forms[person] = {f: (prob+boost)/new_total if f==other_form else prob/new_total for f, prob in prob_dict.items()}
        # print(self.forms[person])
        self.model.n_total_forgets+=1