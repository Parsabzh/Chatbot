non_touristic_food = ['romanian', 'dutch', 'persian', 'british']

inference_table = {'touristic':
                       {'true':
                            {'pricerange': 'cheap',
                             'foodquality': 'good'},
                        'false':
                            {'food': non_touristic_food},
                        'dialogue':
                            'The restaurant is touristic, because it serves cheap and good touristic food'},
                   'romantic':
                       {'true':
                            {'lengthofstay': 'long'},
                        'false':
                            {'crowdedness': 'busy'},
                        'dialogue': 'The restaurant is romantic, because it allows you to stay long and is not busy'},
                   'children':
                       {'true':
                            {},
                        'false':
                            {'lengthofstay': 'long'},
                        'dialogue': 'The restaurant is suitable for children, because it your stay will not be too long'},
                   'sit':
                       {'true':
                            {'crowdedness': 'busy'},
                        'false':
                            {},
                        'dialogue': 'The restaurant has assigned seating, because it is busy'}
                   }


def infer_preferences(suggestions,
                      preferences):  # inferred preferences are based upon the 'conditions' attribute in the
    # restaurant dictionary
    if not preferences.get('condition', False):
        return suggestions


    # satisfy all the inference in the table
    for restaurant in suggestions:  # check for each restaurant
        remove = False
        conditions = preferences['condition']
        if isinstance(preferences['condition'], str):
            conditions = [preferences['condition']]
        for condition in conditions:  # check for each condition
            for key, value in inference_table[condition]['true'].items():  # enforce positive inference
                if restaurant[key] not in value:
                    remove = True
            for key, value in inference_table[condition]['false'].items():  # enforce negative inference
                if restaurant[key] in value:
                    remove = True
        if remove:
            suggestions.remove(restaurant)

    return suggestions


def inferred_dialogue(conditions=None): #return a string containing the inferred restaurant
    dialogue = ""
    if len(conditions) == 0:
        return ""
    for condition in conditions:
        dialogue += inference_table[condition]['dialogue'] + "\n"
