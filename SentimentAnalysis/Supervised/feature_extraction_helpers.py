from emot.emo_unicode import UNICODE_EMO, EMOTICONS
def extract_emojis_semantic(tokens):
    for emot in UNICODE_EMO:
        for token in tokens:
            if token == emot:
                meaning = token.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
                tokens.append(meaning)
                tokens.remove(emot)
    return tokens


def find_number_of_emojis(tokens, emojis):
    count = 0
    for token in tokens:
        if token in emojis:
            count += 1
    return count
