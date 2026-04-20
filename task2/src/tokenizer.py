import re

def parse_segments(file_path):
    segments = []

    pattern = r"\[(.*?)\]\s*\((.*?)\)\s*(.*)"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if not match:
            continue

        lang_tag, timestamp, text = match.groups()
        lang = "en" if "English" in lang_tag else "hi"

        # clean text
        text_clean = re.sub(r"[^\w\s']", "", text)
        words = re.findall(r"\b\w+(?:'\w+)?\b", text_clean)

        segments.append({
            "lang": lang,
            "timestamp": timestamp,
            "words": words
        })

    return segments