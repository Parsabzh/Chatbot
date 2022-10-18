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

scenarios = [{'name': 'scenario_1', 'content': 'Person is looking for chinese restaurant', 'restaurant': 'charlie chan'},
             {'name': 'scenario_2', 'content': 'Person is looking for indian restaurant', 'restaurant': 'charlie chan'}]

random.shuffle(scenarios)


def run_scenarios(i, TTS):
    for index in range(*i):
        scenario = scenarios[index]
        print(scenario['content'])  # print a random scenario
        print("Do your best to find this person a restaurant! Enter anything to continue")
        input()
        dm = DialogManager(TTS=False)
        dialogue = dm.dialogue
        time = dm.end_time
        restaurant = dm.restaurant
        log_dialogue(dialogue, scenario, time, TTS)


def log_dialogue(dialogue, scenario, time, TTS):
    f = open(directory + "/" + scenario['name'] + "_TTS=" + str(TTS) + ".txt", "w")
    f.write("Time: " + str(time) + "\n")
    for d in dialogue:
        f.write(d)
        f.write("\n")


if identifier == "A":  # Group with normal chat first

    run_scenarios((0,5), False)
    run_scenarios((5,10), True)

if identifier == "B":  # Group with TTS first

    run_scenarios((0,5), True)
    run_scenarios((5,10), False)
