from mesa import Agent

from verb import Verb
from util import choice_prob, update_communicated
from config import RG, logging, BOOST


class Agent(Agent):
    def __init__(self, pos, innovating, model):


        super().__init__(pos, model)
        self.pos = pos

        self.innovating=innovating
        # TODO: later possibly add corpus probabilities
        # TODO: Move initialization outside agent?
        self.verb_concepts = ["a"]
        self.persons = ["1sg", "2sg", "3sg"]
        forms_template_conservating = {"0":1.0, "1": 0.0}
        forms_template_innovating = {"0":0.1, "1": 0.9}
        self.forms = {}
        for c in self.verb_concepts:
            for p in self.persons:
                self.forms[c,p] = forms_template_innovating if innovating else forms_template_conservating

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
        concept = RG.choice(self.verb_concepts)
        self.person_question = RG.choice(self.persons)
        form_question = choice_prob(self.forms[concept, self.person_question])
        self.boost_form(concept, self.person_question, form_question)
        # Add to stats
        update_communicated(form_question, self.person_question, self.innovating, self.model, self)
        

        signal_question = Verb(concept=concept, person=self.person_question, form=form_question)
        return signal_question

    # Methods used when agent listens

    def receive_question_reply(self, signal):
        concept, person_question, form_question = signal.get_content()
        self.boost_form(concept, person_question, form_question)

        if self.model.repeats:
            person_answer = self.question_answer_mapping[person_question]
            if person_answer == person_question:
                # 3sg: instead of using own forms library, just repeat form from question
                form_answer = form_question
            else:
                # Other cases: use form from own library
                form_answer = choice_prob(self.forms[concept, person_answer])
            self.boost_form(concept, person_answer, form_answer)
            # if person_answer == person_question:
            #     # Do extra boost if person is same!
            #     self.boost_form(concept, person_answer, form_answer)
            # Add to stats
            update_communicated(form_answer, person_answer, self.innovating, self.model, self)

            signal_answer = Verb(concept=concept, person=person_answer, form=form_answer)
        else:
            # In no-repeats model, answer with yes/no, so no form used
            signal_answer=None
        return signal_answer
    
    def receive_answer(self, signal):
        concept, person_answer, form_answer = signal.get_content()
        self.boost_form(concept, person_answer, form_answer)
        # if person_answer == self.person_question:
        #     # Do extra boost if person is same!
        #     self.boost_form(concept, person_answer, form_answer)
        
        # Reset internal person question variable
        self.pereson_question = None
    
    def boost_form(self, concept, person, form):
        # print("Old")
        # print(self.forms[concept, person])
        prob_dict = self.forms[concept, person]
        new_total = sum(prob_dict.values()) + BOOST
        # Add BOOST to this form and scale by new total, scale other forms by new total
        self.forms[concept, person] = {f: (prob+BOOST)/new_total if f==form else prob/new_total for f, prob in prob_dict.items()}
        # print("New")
        # print(self.forms[concept, person])


        

