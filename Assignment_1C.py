def infer_preferences(suggestions,
                      preferences):  # inferred preferences are based upon the 'conditions' attribute in the
    # restaurant dictionary
    if not preferences.get('condition', False):
        return suggestions
    print(suggestions)
    non_touristic_food = ['romanian', 'dutch', 'persian', 'british']
    print(preferences)

    inference_table = {'touristic':
                           {'true':
                                {'pricerange': 'cheap',
                                 'foodquality': 'good'},
                            'false':
                                {'food': non_touristic_food}},
                       'romantic':
                           {'true':
                                {'lengthofstay': 'long'},
                            'false':
                                {'crowdedness': 'busy'}},
                       'children':
                           {'true':
                                {},
                            'false':
                                {'lengthofstay': 'long'}},
                       'sit':
                           {'true':
                                {'crowdedness': 'busy'},
                            'false':
                                {}}
                       }

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
