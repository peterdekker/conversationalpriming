
class Verb():
    def __init__(self, concept, person, form):
        self.concept = concept
        self.person = person
        self.form = form
    
    def get_content(self):
        return self.concept, self.person, self.form