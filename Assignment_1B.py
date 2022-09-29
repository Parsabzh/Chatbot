from ast import Delete
from operator import index
from re import M, U
import Levenshtein as ls
from Levenshtein import distance as lev
from NN1 import NeuralNet as neural_net_classifier, create_dataframe
from Assignment_1C import infer_preferences
import pandas as pd

restaurant_data = pd.read_csv("Data/restaurant_info.csv", sep=';')[0:]
restaurants = restaurant_data.to_dict('records')


class DialogManager:
    def __init__(self):
        self.state = 'start'
        print(
            'Hello, welcome to the Restaurant Recommendation System. You can ask for restaurants by area, price range, or foodtype. How may I help you?')
        self.preferences = {'area': '', 'food': '', 'pricerange': ''}
        self.dialogue_act = None
        # dt = create_dataframe()
        self.nn = neural_net_classifier()
        self.restaurant = None
        self.loop()

    def state_transition(self, utterance):
        dialogue_act = None
        speech_act = self.nn.predict(utterance)
        print(speech_act)
        
        # when user input is inform extract new preferences and suggest restaurant
        if speech_act == 'inform' or speech_act == 'request':
            self.preferences = self.preferences | extract_preferences(
                utterance)

            if self.preferences['area'] != '' and self.preferences['food'] != '' and self.preferences[
                    'pricerange'] != '':

                # When the preferences are know go to state suggest restaurant, it can be found or not found
                self.state = "suggest_restaurant"
                suggestions, min_score = restaurant_suggestion(
                    self.preferences)
                self.restaurant = suggestions[0]

                # We look at the distance score if its not zero suggest next best restaurant
                rst = self.restaurant
                if min_score != 0:
                    dialogue_act = "Im sorry there is no " + self.preferences[
                        'pricerange'] + ' ' + self.preferences['food'] + ' restaurant on the ' + self.preferences['area'] + ' side of town.'
                    dialogue_act += "\n But we have an alternative. Is the " + str(rst['food']) + ' restaurant "' + str(rst['restaurantname']) + '" on the ' + str(
                        rst['area']) + ' part of town with a ' + rst['pricerange'] + " price range ok?"
                    self.state = 'after_suggestion'
                else:

                    # Give perfect match
                    dialogue_act = "Is " + str(rst['restaurantname']) + 'on the' + str(
                        rst['area']) + ' part of town with a ' + rst['pricerange'] + " price range ok?"
                    self.state = 'after_suggestion'
            else:

                # When some of the preferences are empty fill them in by going to the request state
                for key, value in self.preferences.items():
                    if value == '':
                        self.state = 'request_' + str(key)
                        break
            # Request states to ask about unknown preferences
            if self.state == 'request_area':
                dialogue_act = 'In which area would you like to eat?'
            if self.state == 'request_food':
                dialogue_act = 'What kind of food would you like to eat?'
            if self.state == 'request_pricerange':
                dialogue_act = 'What pricerange does the food have to be?'

        # After restaurant suggestion affirm, deny and request dialogue acts
        if self.state == 'after_suggestion':
            if speech_act == 'affirm':
                self.state = 'end'
                dialogue_act = "Thank you for using the system. Goodbye!"
            if speech_act == 'deny':
                dialogue_act = "what would you like instead?"
                self.state = 'suggest_restaurant'

        # After goodbye utterance go to end state
        if speech_act == 'goodbye':
            self.state = 'end'
            dialogue_act = "Thank you for using the system. Goodbye!"

        return dialogue_act

    def loop(self):
        while self.state != 'end':
            utterance = input().lower()
            dialogue_act = self.state_transition(utterance)
            print(dialogue_act)


def extract_preferences(utterance):
    data = {"area": ['west', 'east', 'south', 'north', 'center'],
            "food": ['italian', 'romanian', 'dutch', 'persian', 'american', 'chinese', 'british', 'greece', 'world',
                     'swedish', 'international', 'catalan', 'cuban', 'tuscan'],
            "condition": ['busy', 'romantic', 'children', 'sit'],
            "pricerange": ['cheap', 'expensive', 'moderate']}

    words = utterance.split()
    preferences = {}
    for word in words:

        for key, val_list in list(data.items()):
            if word in val_list:
                preferences.update({key: word})
                del data[key]
            if word == 'any':
                preferences.update({'area': 'any'})
                del data['area']
        n = len(word)
        for key, val_list in data.items():
            for val in val_list:
                m = lev(word, val)
                if m < n:
                    n = m

                    # Lev score smaller than 3
                    if n < 3:
                        preferences.update({key: val})
    return preferences


# f = extract_preferences(' i want to go cheap restuarant in south part of the town')


def restaurant_suggestion(preferences):
    scores = []
    for restaurant in restaurants:
        score = 0
        for preference, value in preferences.items():
            if value == "any" or preference == 'condition':
                continue
            distance = ls.distance(str(value), str(restaurant[preference]))
            score += distance
        scores.append((score, restaurant))
    scores.sort(key=lambda x: x[0])

    min_score = scores[0][0]
    _, suggestions = zip(*scores[:10]) #suggestions are the top 10 scoring restaurants

    inferred_suggestions = infer_preferences(suggestions, preferences)

    # return list with highest scoring restaurants, sorted from best to worst
    return inferred_suggestions, min_score

#print(restaurant_suggestion({'pricerange': 'expenove', 'food': 'spenush'}))
dialogue = DialogManager()
