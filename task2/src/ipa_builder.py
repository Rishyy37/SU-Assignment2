from tqdm import tqdm
from src.g2p_converter import hinglish_g2p


def build_timestamped_ipa(segments, output_path):
    output_lines = []

    for seg in tqdm(segments):
        timestamp = seg["timestamp"]
        words = seg["words"]
        lang = seg["lang"]

        ipa_words = []

        for w in words:
            ipa = hinglish_g2p(w.lower(), lang)
            ipa_words.append(ipa)

        ipa_line = f"[{timestamp}] " + " | ".join(ipa_words)
        output_lines.append(ipa_line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    return output_lines