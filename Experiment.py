from Assignment_1B import DialogManager

print("Hello participant! Welcome to our experiment. Please read the instructions carefully before starting the experiment.")

print("What is your name?")
name = input()
print("What is your age?")
age = input()
print("What is your identifier?")
identifier = input()

scenarios = {}

if identifier == "A": # Group with normal chat first
    for i in range(5):
        print(scenarios) # print a random scenario
        dialogue = DialogManager(TTS=False)

        session_log[dialogue_count] = {"dialogue": dialogue, "scenario": scenarios}

    for i in range(5):
        print(scenarios)
        dialogue = DialogManager(TTS=True)

if identifier == "B":
    for i in range(5):
        print(scenarios)
        dialogue = DialogManager(TTS=True)

    for i in range(5):
        print(scenarios)
        dialogue = DialogManager(TTS=False)

def log_dialogue(dialogue, scenario, identifier):
    with open("dialogue_log.csv", "a") as f:
        f.write(identifier + "," + scenario + "," + dialogue)
