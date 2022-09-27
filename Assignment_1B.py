import Levenshtein as ls
from NN1 import NeuralNet as neural_net_classifier, create_dataframe
import pandas as pd

restaurant_data = pd.read_csv("Data/restaurant_info.csv")[0:]
restaurants = restaurant_data.to_dict('records')


class DialogManager:
    def __init__(self):
        self.state = 'start'
        self.preferences = {}
        self.dialogue_act = None
        self.nn = neural_net_classifier().load_model()
        self.restaurant = None

        self.loop(self)

    def state_transition(self, state, utterance):

        if state == 'inform' or state == 'request':
            extract_preferences(utterance)

        dialogue_class = self.nn.predict(utterance)
        if state == 'goodbye':
            self.state = 'end'
            dialogue_act = "Thank you for using the system. Goodbye!"
        return state, dialogue_act

    def loop(self):
        while self.state != 'end':
            utterance = input().lower()
            state, dialogue_act = self.state_transition(state, utterance)
            print(dialogue_act)


def extract_preferences(utterance):
    preferences = {}
    return preferences


def restaurant_suggestion(preferences):
    scores = []
    for restaurant in restaurants:
        score = 0
        for preference, value in preferences.items():
            distance = ls.distance(value, restaurant[preference])
            score += distance
        scores.append((score, restaurant))
    return min(scores, key=lambda x: x[0])[1]  # return highest scoring restaurant


print(restaurant_suggestion({'pricerange': 'expenove', 'food': 'spenush'}))
