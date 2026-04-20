from g2p_en import G2p
import json

g2p = G2p()

# Load Hindi IPA mapping
with open("models/guj_ipa_map.json", "r", encoding="utf-8") as f:
    hindi_map = json.load(f)


def clean_ipa(ipa_list):
    return " ".join([p for p in ipa_list if p.strip() != ""])


def hinglish_g2p(word, lang):
    if lang == "en":
        try:
            return clean_ipa(g2p(word))
        except:
            return word

    elif lang == "hi":
        return hindi_map.get(word, word)

    return word