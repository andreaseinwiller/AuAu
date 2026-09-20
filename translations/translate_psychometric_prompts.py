import os

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.getenv("ALT_BASE_URL"),
    api_key=os.getenv("ALT_API_KEY"),
)

SYSTEM_PROMPT = """You are a professional localization assistant.

Translate the English text to Simplified Chinese (zh-Hans).

Rules:
- Preserve Jinja2 syntax EXACTLY:
  - {{ ... }}
  - {% ... %}
  - {# ... #}
- Do NOT translate variable names, function names, or control keywords.
- Do NOT add explanations or comments.
- Treat the input as source code with embedded human-readable text.
- If unsure whether something is code or text, DO NOT translate it.

Output ONLY the translated template.
"""


def translate_with_llm(text: str) -> str:
    response = client.responses.create(
        model="qwen3-next-80b-a3b-instruct",  # or whatever model your endpoint exposes
        temperature=0,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )

    # Extract plain text output
    return response.output_text.strip()


def translate_file(en_path: Path) -> None:
    zh_path = en_path.with_name("zh.j2")

    if zh_path.exists():
        print(f"Skipping (already exists): {zh_path}")
        return

    source_text = en_path.read_text(encoding="utf-8")

    translated_text = translate_with_llm(source_text)

    zh_path.write_text(translated_text, encoding="utf-8")
    print(f"Created: {zh_path}")


def main() -> None:
    root = Path("resources/input/datasets/prompts")

    for en_file in root.rglob("en.j2"):
        translate_file(en_file)


if __name__ == "__main__":
    main()
