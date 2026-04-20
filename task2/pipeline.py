from src.tokenizer import parse_segments
from src.ipa_builder import build_timestamped_ipa
from src.translator import translate_tokens


INPUT_FILE = "data/input_transcript.txt"
IPA_OUTPUT = "data/ipa_output.txt"
LRL_OUTPUT = "data/translated_lrl.txt"


def main():
    print("Step 1: Parsing segments...")
    segments = parse_segments(INPUT_FILE)

    print("Step 2: Generating timestamped IPA...")
    ipa_lines = build_timestamped_ipa(segments, IPA_OUTPUT)

    print("Step 3: Flattening tokens for translation...")
    tokens = []
    for seg in segments:
        for w in seg["words"]:
            tokens.append((w.lower(), seg["lang"]))

    print("Step 4: Translating to LRL...")
    translated_text = translate_tokens(tokens, LRL_OUTPUT)

    print("\n✅ Task 2 Completed!")
    print(f"IPA saved at: {IPA_OUTPUT}")
    print(f"LRL translation saved at: {LRL_OUTPUT}")


if __name__ == "__main__":
    main()