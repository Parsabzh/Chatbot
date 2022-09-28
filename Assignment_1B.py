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
        print('Hello, welcome to the Restaurant Recommendation System. You can ask for restaurants by area, price range, or foodtype. How may I help you?')
        self.preferences = {'area' : '', 'food' : '', 'pricerange' : ''}
        self.dialogue_act = None
        # dt = create_dataframe()
        self.nn = neural_net_classifier().load_model()
        self.restaurant = None
        self.loop()

    def state_transition(self, state, utterance):

        speech_act = self.nn.predict(utterance)

        if speech_act == 'inform' or speech_act == 'request':
            self.extract_preferences(utterance)


        for preference, value in self.preferences.items():
            if (value == '' and preference in ['area','food','pricerange']):
                state = 'request_' + str(preference)
                break
            
    
        if self.preferences['area'] != '' and self.preferences['food'] != '' and self.preferences['pricerange'] != '':
            state = "suggest_restaurant"
            self.restaurant = restaurant_suggestion(self.preferences)
            rst = self.restaurant
            dialogue_act = "is " + str(rst['restaurantname']) + 'on the' + str(rst['area']) + ' part of town' + " ok?"


        if state == 'request_area':
            dialogue_act = 'In which area would you like to eat?'

        if state == 'request_food':
            dialogue_act = 'What kind of food would you like to eat?'

        if state == 'request_pricerange':
            dialogue_act = 'What pricerange does the food have to be?'

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
