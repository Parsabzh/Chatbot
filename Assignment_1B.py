from ast import Delete
from operator import index
from re import M
import Levenshtein as ls
from Levenshtein import distance as lev
from NN1 import NeuralNet as neural_net_classifier, create_dataframe
import pandas as pd

restaurant_data = pd.read_csv("Data/restaurant_info.csv")[0:]
restaurants = restaurant_data.to_dict('records')


class DialogManager:
    def __init__(self):
        self.state = 'start'
        self.preferences = {}
        self.dialogue_act = None
        # dt = create_dataframe()
        self.nn = neural_net_classifier().load_model()
        self.restaurant = None

        self.loop()

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
    data={"location":['west','east','south','north','center'],
    "food":['italian','romanian','dutch','persian','american','chines','british','greece','world','swedish','international','catalan','cuban','tuscan'],
    "condition":['busy','romantic','children','sit'],
    "price":['cheap','expensive','moderate']}
    
    words=utterance.split()
    preferences = {}
    for word in words:
        
        for key,val_list in list(data.items()):
           if word in val_list:
                preferences.update({key:word})
                del data[key]
           if word=='any':
                preferences.update({'location':'any'})   
                del data['location'] 
        n=len(word)
        for key,val_list in data.items():
            for val in val_list:
                    m= lev(word,val)
                    if m<n:
                        n=m
                        if n<4:    
                            preferences.update({key:val})         
            
    print(preferences)            
    return preferences

f= extract_preferences(' i want to go cheap restuarant in south part of the town')
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
