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
identifier = input()

session_name = "Data/session_" + name + "_" + age + "_" + identifier
parent_directory = os.getcwd()
directory = os.path.join(parent_directory, session_name)
os.mkdir(directory)

scenarios = [{'name' : 'scenario_1', 'content': 'blablabla'}]


def log_dialogue(dialogue, scenario_name, time, TTS):
    f = open(directory + "/" + scenario_name + "_TTS:_" + str(TTS) + ".txt", "w")
    f.write(str(time))
    f.write(str(dialogue))


identifier = identifier.upper()

if identifier == "A":  # Group with normal chat first
    for i in range(5):
        scenario = random.sample(scenarios, 1)[0]
        print(scenario['content'])  # print a random scenario
        dm = DialogManager(TTS=False)
        dialogue = DialogManager.dialogue
        time = DialogManager.end_time
        log_dialogue(dialogue, scenario['name'], time, TTS=False)

    for i in range(5):
        scenario = random.sample(scenarios, 1)[0]
        print(scenario['content'])  # print a random scenario
        dm = DialogManager(TTS=True)
        dialogue = DialogManager.dialogue
        time = DialogManager.end_time
        log_dialogue(dialogue, scenario['name'], time, TTS=True)

if identifier == "B":  # Group with TTS first
    for i in range(5):
        scenario = random.sample(scenarios, 1)[0]
        print(scenario['content'])  # print a random scenario
        dm = DialogManager(TTS=True)
        dialogue = DialogManager.dialogue
        time = DialogManager.end_time
        log_dialogue(dialogue, scenario['name'], time, TTS=True)

    for i in range(5):
        scenario = random.sample(scenarios, 1)[0]
        print(scenario['content'])  # print a random scenario
        dm = DialogManager(TTS=False)
        dialogue = DialogManager.dialogue
        time = DialogManager.end_time
        log_dialogue(dialogue, scenario['name'], time, TTS=False)
