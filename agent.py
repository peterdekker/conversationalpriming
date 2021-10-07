from mesa import Agent

from verb import Verb
from util import choice_prob
from config import RG, logging, BOOST


class Agent(Agent):
    def __init__(self, pos, model):
        '''
         Create a new speech agent.

         Args:
            pos: Agent initial location.
            model: Model in which agent is located.
            init: Initialization mode of forms and affixes: random or data
        '''
        super().__init__(pos, model)
        self.pos = pos
        self.colours = "hsl(250,80%,50%)", "hsl(250,80%,50%)"

        # TODO: later possibly add corpus probabilities
        self.verb_concepts = ["a", "b"]
        self.persons = ["1sg", "2sg", "3sg"]
        forms_template = {"1":0.9, "2": 0.1}
        self.forms = {}
        for c in self.verb_concepts:
            for p in self.persons:
                self.forms[c,p] = forms_template

        self.question_answer_mapping = {"1sg":"2sg", "2sg":"1sg", "3sg":"3sg"}

    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        listener = RG.choice(self.model.agents)

        question = self.send_question()
        answer = listener.receive_question(question)
        self.receive_answer(answer)


    def send_question(self):
        # Send polar question, represented by verb: concept, person and form
        concept = RG.choice(self.verb_concepts)
        person_question = RG.choice(self.persons)
        form_question = choice_prob(self.forms[concept, person_question])
        self.boost_form(concept, person_question, form_question)

        signal_question = Verb(concept=concept, person=person_question, form=form_question)
        return signal_question

    # Methods used when agent listens

    def receive_question(self, signal):
        concept, person_question, form_question = signal.get_content()
        self.boost_form(concept, person_question, form_question)

        person_answer = self.question_answer_mapping[person_question]
        form_answer = choice_prob(self.forms[concept, person_answer])
        self.boost_form(concept, person_answer, form_answer)

        signal_answer = Verb(concept=concept, person=person_answer, form=form_answer)
        return signal_answer
    
    def receive_answer(self, signal):
        concept, person_answer, form_answer = signal.get_content()
        self.boost_form(concept, person_answer, form_answer)
    
    def boost_form(self, concept, person, form):
        print("Old")
        print(self.forms[concept, person])
        prob_dict = self.forms[concept, person]
        new_total = sum(prob_dict.values()) + BOOST
        # Add BOOST to this form and scale by new total, scale other forms by new total
        self.forms[concept, person] = {f: (prob+BOOST)/new_total if f==form else prob/new_total for f, prob in prob_dict.items()}
        print("New")
        print(self.forms[concept, person])


        

