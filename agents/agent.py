import mesa

from agents.verb import Verb
from agents.util import choice_prob, update_communicated
from agents.config import RG, logging, PERSONS, INNOVATIVE_FORM
import numpy as np


class Agent(mesa.Agent):
    def __init__(self, uid, innovator, prop_innovative_forms, model):


        super().__init__(uid, model)

        self.uid = uid
        self.innovator = innovator
        self.persons = PERSONS
        self.forms = {}
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
        if not self.model.use_grid: # default: use network
            neighbors_nodes = self.model.grid.get_neighborhood(self.uid, include_center=False)
            neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
            listener = RG.choice(neighbors)
        else: # obsolete: use grid
             listener = RG.choice([a for a in self.model.agents if a.uid != self.uid])
        
        for i in range(self.model.n_interactions_interlocutor):
            question = self.create_question()
            answer = listener.receive_question_reply(question)
            if answer is not None:
                self.receive_answer(answer)
            
            # After every interaction, forget token
            # Forget one form for every person. This makes frequency assymetry between persons relevant.
            if self.model.decay > 0.0:
                for person in self.persons:
                    # Simulate sampling from list, by sampling with inverse probability of form
                    # So if 0.9 probability for token 1: 0.1 chance to forget token 1 and 0.9 chance to forget token 0
                    sample_token = min(self.forms[person], key=self.forms[person].get) # choice_prob(self.forms[person], inverse=True) # 
                    self.forget_form(person, sample_token)


    def create_question(self):
        self.person_question = RG.choice(self.persons, p=self.person_weights)
        form_question = choice_prob(self.forms[self.person_question])
        self.increase_form(self.person_question, form_question)
        # Add to stats
        update_communicated(form_question, self.person_question, self.innovator, self.model, self)
        

        signal_question = Verb(person=self.person_question, form=form_question)
        return signal_question

    # Methods used when agent listens

    def receive_question_reply(self, signal):
        person_question, form_question = signal.get_content()
        if not (self.model.innovator_only_increase_production and self.innovator):
            self.increase_form(person_question, form_question)

        if self.model.repeats:
            person_answer = self.question_answer_mapping[person_question]
            if person_answer == person_question and RG.random() < self.model.conversational_priming_prob and not (self.model.innovator_no_conversational_priming and self.innovator):
                # 3sg: instead of using own forms library, just repeat form from question
                form_answer = form_question
                # If double increase parameter is on, give double increase in case of conversational priming
                self.increase_form(person_answer, form_answer, self.model.double_increase_conv_priming_production)
            else:
                # Other cases: use form from own library
                form_answer = choice_prob(self.forms[person_answer])
                self.increase_form(person_answer, form_answer)
            # Add to stats
            update_communicated(form_answer, person_answer, self.innovator, self.model, self)

            signal_answer = Verb(person=person_answer, form=form_answer)
        else:
            # In no-repeats model, answer with yes/no, so no form used
            signal_answer=None
        return signal_answer
    
    def receive_answer(self, signal):
        person_answer, form_answer = signal.get_content()
        if not (self.model.innovator_only_increase_production and self.innovator):
            self.increase_form(person_answer, form_answer)

        # Reset internal person question variable
        self.person_question = None
    
    def increase_form(self, person, form, double_increase_conv_priming_production=False):
        # if (self.innovator and form != INNOVATIVE_FORM):
        #     return
        SURPRISAL_THRESHOLD = 1000000
        prob_dict = self.forms[person]
        if form == INNOVATIVE_FORM:
            increase = self.model.increase_innovative
        else:
            increase = self.model.increase_conservative
        if double_increase_conv_priming_production:
            increase *= 2
        if self.model.surprisal:
            #surprisal = min(-np.log2(prob_dict[form]), SURPRISAL_THRESHOLD)
            surprisal = -np.log2(prob_dict[form])
            increase = increase * surprisal
        if self.model.entropy and not self.model.surprisal:
            px = prob_dict[form]
            # surprisal = min(-np.log2(px), SURPRISAL_THRESHOLD)
            surprisal = -np.log2(px)
            entropy = px * surprisal
            increase = increase * 2 * entropy
        new_total = 1.0 + increase
        # Add INCREASE to this form and scale by new total, scale other forms by new total
        self.forms[person] = {f: (prob+increase)/new_total if f==form else prob/new_total for f, prob in prob_dict.items()}

        # Counter for diagnostic purposes
        self.model.n_total_increases += 1
    
    def forget_form(self, person, form):
        prob_dict = self.forms[person]
        increase = self.model.decay
        new_total = 1.0 + increase
        # Forget this form by increaseing other form
        other_form = "1" if form=="0" else "0"
        # print(f"Perform forget for {form} by increaseing {other_form}")
        # print(self.forms[person])
        # Add INCREASE to this form and scale by new total, scale other forms by new total
        self.forms[person] = {f: (prob+increase)/new_total if f==other_form else prob/new_total for f, prob in prob_dict.items()}
        # print(self.forms[person])
        self.model.n_total_forgets+=1