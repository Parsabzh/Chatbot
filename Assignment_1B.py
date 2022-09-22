import Levenshtein as ls

state = 'start'


def state_transition(state, utterance):

    if state == 'goodbye':
        state = 'end'
        dialogue_act = "Thank you for using the system. Goodbye!"
    return state, dialogue_act


def restaurant_suggestions(preferences):
    suggestions = []
    return suggestions


while state != 'end':
    utterance = input().lower()
    state, dialogue_act = state_transition(state, utterance)
    print(dialogue_act)
