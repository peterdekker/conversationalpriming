
class Verb():
    def __init__(self, person, form):
        self.person = person
        self.form = form
    
    def get_content(self):
        return self.person, self.form