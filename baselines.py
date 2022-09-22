# return the most frequent class as baseline
def baseline_classification():
    baseline = 'inform' #y.mode()[0]
    return baseline


# classify an utterance based on simple keywords
def baseline_classification2(sentence):
    keywords = {
        'thankyou': ['thank', 'thank', 'thanks', 'thank', 'thanks', 'you', 'goodbye', 'bye','good'],
        'ack': ['okay', 'ok', 'fine', 'great', 'cool'],
        'affirm': ['yes', 'yeah', 'yep', 'yup', 'sure', 'of course', 'right', 'correct', 'indeed', 'absolutely'],
        'bye': ['goodbye', 'bye', 'see you', 'later', 'soon'],
        'confirm': ['is', 'it'],
        'deny': ['dont', 'do not'],
        'hello': ['hi', 'hello', 'hey', 'greetings', 'howdy'],
        'inform': ['looking', 'for', 'want', 'need', 'searching', 'care'],
        'negate': ['no', 'nope', 'nah'],
        'null': ['cough', 'hm', 'unintelligible', 'noise'],
        'repeat': ['can', 'repeat', 'that'],
        'reqalts': ['how', 'about', 'other', 'options', 'else', 'another', 'alternatives', 'else'],
        'reqmore': ['more'],
        'request': ['what', 'where', 'when', 'who', 'which', 'how', 'why', 'whats', 'wheres', 'whens', 'whos', 'whens',
                    'hows', 'whys', 'phone', 'number', 'price', 'range', 'address', 'can', 'type', 'code', 'post'],
        'restart': ['start', 'over', 'restart'],
    }
    counts = {}
    for word in sentence.split():
        for key in keywords:
            if word in keywords[key]: #assign scores to each keyword 'bucket'
                counts[key] = counts.get(key, 0) + 1

    if counts != {}:
        return max(counts, key=counts.get)
    else:
        return baseline_classification()

