from mesa import Agent

from config import RG, logging


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

    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        # If radius==MAX_RADIUS, every agent speaks with every other, so random mixing
        neighbors = self.model.grid.get_neighbors(self.pos, True, False, self.model.radius)
        listener = RG.choice(neighbors)
        self.speak(listener)

    # Methods used when agent speaks

    def speak(self, listener):
        '''
         Speak to other agent

         Args:
            listener: agent to speak to
        '''
        pass

    # Methods used when agent listens

    def listen(self, signal):
        '''
         Agent listens to signal sent by speaker

         Args:
            signal: received signal

         Returns:
            message: concept which listener thinks is closest to heard signal
        '''
        pass
        return 0

    def receive_feedback(self, feedback_speaker):
        '''
        Listening agent receives concept meant by speaking agent,
        and updates its language table

        Args:
            feedback_speaker: feedback from the speaker
        '''

        pass

