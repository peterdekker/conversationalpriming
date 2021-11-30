from mesa import Agent

from verb import Verb
from util import choice_prob, update_communicated
from config import RG, logging, SMOOTHING_SURPRISAL, PERSONS
import numpy as np


class Agent(Agent):
    def __init__(self, pos, innovating, prop_innovative_forms, model):


        super().__init__(pos, model)
        self.pos = pos
        self.innovating = innovating
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


    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        listener = RG.choice(self.model.agents)

        question = self.create_question()
        answer = listener.receive_question_reply(question)
        if answer is not None:
            self.receive_answer(answer)


    def create_question(self):
        # Send polar question, represented by verb: concept, person and form
        #concept = RG.choice(self.verb_concepts)
        self.person_question = RG.choice(self.persons)
        form_question = choice_prob(self.forms[self.person_question])
        self.boost_form(self.person_question, form_question)
        # Add to stats
        update_communicated(form_question, self.person_question, self.innovating, self.model, self)
        

        signal_question = Verb(person=self.person_question, form=form_question)
        return signal_question

    # Methods used when agent listens

    def receive_question_reply(self, signal):
        person_question, form_question = signal.get_content()
        self.boost_form(person_question, form_question)

        if self.model.repeats:
            person_answer = self.question_answer_mapping[person_question]
            if person_answer == person_question:
                # 3sg: instead of using own forms library, just repeat form from question
                form_answer = form_question
            else:
                # Other cases: use form from own library
                form_answer = choice_prob(self.forms[person_answer])
            self.boost_form(person_answer, form_answer)
            # if person_answer == person_question:
            #     # Do extra boost if person is same!
            #     self.boost_form(concept, person_answer, form_answer)
            # Add to stats
            update_communicated(form_answer, person_answer, self.innovating, self.model, self)

            signal_answer = Verb(person=person_answer, form=form_answer)
        else:
            # In no-repeats model, answer with yes/no, so no form used
            signal_answer=None
        return signal_answer
    
    def receive_answer(self, signal):
        person_answer, form_answer = signal.get_content()
        self.boost_form(person_answer, form_answer)

        # Reset internal person question variable
        self.person_question = None
    
    def boost_form(self, person, form):
        SURPRISAL_THRESHOLD = 1000000
        prob_dict = self.forms[person]
        boost = self.model.boost
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


        

