import Levenshtein as ls
from NN1 import NeuralNet as neural_net_classifier
import pandas as pd
class DialogManager:
    def __init__(self):
        self.state = 'start'
        self.preferences = {}
        self.dialogue_act = None
        dt = pd.DataFrame(columns=['dialogue', 'uttr'])
        self.nn = neural_net_classifier(dt)

    def state_transition(self, state, utterance):

        if state == 'inform':


        dialogue_class = self.nn.predict(utterance)
        if state == 'goodbye':
            state = 'end'
            dialogue_act = "Thank you for using the system. Goodbye!"
        return state, dialogue_act

    def restaurant_suggestions(self):
        preferences = self.preferences
        suggestions = []
        return suggestions

    while state != 'end':
        utterance = input().lower()
        state, dialogue_act = state_transition(state, utterance)
        print(dialogue_act)



