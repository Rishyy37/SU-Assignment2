import json

# Load dictionary
with open("data/lrl_dictionary.json", "r", encoding="utf-8") as f:
    lrl_dict = json.load(f)


def translate_tokens(tokens, output_path):
    translated = []

    for word, lang in tokens:
        if word in lrl_dict:
            translated.append(lrl_dict[word])
        else:
            translated.append(word)

    final_text = " ".join(translated)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    return final_text