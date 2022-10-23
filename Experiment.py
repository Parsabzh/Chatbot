from Assignment_1B import DialogManager
import os
import random

print(
    "Hello participant! Welcome to our experiment. Please read the instructions carefully before starting the experiment.")

print("What is your name?")
name = input()
print("What is your age?")
age = input()
print("What is your identifier?")
identifier = input().upper()

session_name = "Logs/Group_" + identifier + "_" + name + "_" + age
parent_directory = os.getcwd()
directory = os.path.join(parent_directory, session_name)
os.mkdir(directory)

scenarios = [{'name': 'scenario_1', 'content': 'Person is looking for a chinese restaurant, is living in the east part of town and cannot afford expensive food', 'restaurant': 'shanghai family restaurant'},
             {'name': 'scenario_2', 'content': 'Person is looking for a french restaurant in the north and has a lot of money', 'restaurant': 'two two'},
             {'name': 'scenario_3', 'content': 'Person is looking for a mediocre cheap chinese restaurant in the south ', 'restaurant': 'the missing sock'},
             {'name': 'scenario_4', 'content': 'Person is looking for the phone number of an cheap vietnamese restaurant in the west that is bad', 'restaurant': 'thanh binh'},
             {'name': 'scenario_5', 'content': 'Person is looking for a moderately priced romanian restaurant and its address', 'restaurant': 'the gardenia'},
             {'name': 'scenario_6', 'content': 'Person is looking for and expensive mediterranean restaurant that is busy in the centre', 'restaurant': 'shiraz'},
             {'name': 'scenario_7', 'content': 'Person is looking for an expensive gatrobup in the east part of town', 'restaurant': 'cocum'},
             {'name': 'scenario_8', 'content': 'Person is looking for a moderatly priced british restaurant that is busy and its address', 'restaurant': 'cotto'},
             {'name': 'scenario_9', 'content': 'Person is looking for the phone number of a cheap indian restaurant in the centre', 'restaurant': 'mahal of cambridge'},
             {'name': 'scenario_10', 'content': 'Person is looking for a traditional cheap restaurant in the north part of town', 'restaurant': 'vinci pizzeria'}]

random.shuffle(scenarios)


def run_scenarios(i, TTS):
    for index in range(*i):
        scenario = scenarios[index]
        print(scenario['content'])  # print a random scenario
        print("Do your best to find this person a restaurant! Enter anything to continue")
        input()
        dm = DialogManager(TTS=TTS)
        dialogue = dm.dialogue
        time = dm.end_time
        restaurant = dm.restaurant
        log_dialogue(dialogue, scenario, time, TTS)


def log_dialogue(dialogue, scenario, time, TTS):
    f = open(directory + "/" + scenario['name'] + "_TTS=" + str(TTS) + ".txt", "w")
    f.write("Time: " + str(time) + "\n")
    f.write("Complexity: " + str(len(dialogue)) + "\n")
    for d in dialogue:
        f.write(d)
        f.write("\n")



if identifier == "A":  # Group with normal chat first

    run_scenarios((0,5), False)
    run_scenarios((5,10), True)

if identifier == "B":  # Group with TTS first

    run_scenarios((0,5), True)
    run_scenarios((5,10), False)
