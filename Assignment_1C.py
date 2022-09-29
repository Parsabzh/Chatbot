def infer_preferences(suggestions, preferences):
    if not preferences.get('condition', False):
        return suggestions
    non_touristic_food = ['romanian', 'dutch', 'persian', 'british']

    def touristic_inference(restaurant):  # returns True if the restaurant does not meet the criteria set by inference
        if any([
            restaurant['food'] in non_touristic_food,
            restaurant['pricerange'] != 'cheap',
            restaurant['foodquality'] != 'good'
        ]):
            return True

    for condition in preferences['condition']:
        for restaurant in suggestions:
            # for a restaurant to still be suggested it needs to pass the conditions
            match condition:
                case 'touristic':
                    if touristic_inference(restaurant):
                        del suggestions[restaurant]
                case 'sit':
                    if restaurant['crowdedness'] == 'busy':
                        del suggestions[restaurant]
                case 'romantic':
                    if restaurant['crowdedness'] == 'busy' or restaurant['lengthofstay'] != 'longstay':
                        del suggestions[restaurant]
                case 'children':
                    if restaurant['lengthofstay'] == 'long':
                        del suggestions[restaurant]
    return suggestions

