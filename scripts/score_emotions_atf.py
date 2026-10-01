"""
Final emotion-scoring script used in the paper (ATF "Level 2" prompt).

Each Reddit post is sent to DeepSeek V3.2 (deepseek-chat) with a prompt built on the
Appraisal Tendency Framework: every emotion is defined by its valence, its arousal and the
trading behaviour it is expected to trigger. The model returns six intensity scores in [0, 1].

The full sample was split into two halves (part_A_input.csv / part_B_input.csv) and scored
in parallel. The script writes results line by line and skips IDs that are already in the
output file, so a long run can be stopped and resumed.

Usage:
    export DEEPSEEK_API_KEY=...
    python score_emotions_atf.py
"""

import os
import json
import time
import re

import pandas as pd

# ============================================
# CONFIGURATION
# ============================================
API_KEY = os.environ["DEEPSEEK_API_KEY"]  # never hard-code the key
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# Switch to "part_B_input.csv" to score the second half
INPUT_FILE = "part_A_input.csv"

TEXT_COLUMN = "body"
ID_COLUMN = "row_id"

# e.g. part_A_level2_results.csv
OUTPUT_FILE = INPUT_FILE.replace("_input.csv", "_level2_results.csv")

SLEEP_SEC = 0.5
MAX_RETRIES = 3

EMOTIONS = ["joy", "tru", "fea", "sur", "sad", "ang"]


# ============================================
# PROMPT: LEVEL 2 (ATF explicit)
# ============================================
def prompt_level2(body: str) -> str:
    return f"""You are a financial sentiment analyst applying the Appraisal Tendency Framework (ATF)
to score investor emotions in cryptocurrency Reddit posts.

The ATF identifies two key dimensions:
  VALENCE: positive vs negative emotional direction
  AROUSAL: high (urgent, immediate action) vs low (passive, gradual response)

Post: "{body}"

Score the following six emotions (0.0 = absent, 1.0 = dominant):

NEGATIVE HIGH-AROUSAL:
  fea (Fear): High uncertainty + low control → triggers urgent capital flight, panic selling
  ang (Anger): High certainty + blame attribution → triggers contrarian buying, NOT flight

NEGATIVE LOW-AROUSAL:
  sad (Sadness): Low arousal + passivity → triggers gradual withdrawal, NOT urgent selling

NEUTRAL HIGH-AROUSAL:
  sur (Surprise): Unexpected information shock, NO directional valence → forced reassessment

POSITIVE HIGH-AROUSAL:
  joy (Joy): High certainty + positive outlook → triggers momentum chasing, FOMO buying

POSITIVE LOW-AROUSAL:
  tru (Trust): Stable certainty + confidence → triggers passive holding, "diamond hands"

CRITICAL: Fear and Anger are BOTH negative but have OPPOSITE market effects.
Score them independently based on whether the post shows UNCERTAINTY (fear) or BLAME/CERTAINTY (anger).

Return JSON ONLY with no explanation:
{{"joy": 0.0, "tru": 0.0, "fea": 0.0, "sur": 0.0, "sad": 0.0, "ang": 0.0}}"""


# ============================================
# API CALLER
# ============================================
def call_deepseek(client, prompt: str) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()

            # pull the JSON object out of the reply
            json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if all(e in parsed for e in EMOTIONS):
                    return parsed

            print(f"  [Attempt {attempt + 1}] JSON parse warning. Response: {text[:50]}...")
            time.sleep(1)

        except Exception as e:
            print(f"  [Attempt {attempt + 1}] API error: {e}")
            time.sleep(2 ** attempt)

    return None


# ============================================
# MAIN
# ============================================
def main():
    from openai import OpenAI
    from tqdm import tqdm

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 1. load input
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 2. resume: skip IDs that are already scored
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        if ID_COLUMN in existing_df.columns:
            processed_ids = set(existing_df[ID_COLUMN].astype(str))
        print(f"Found existing output, resuming ({len(processed_ids)} rows done).")
    else:
        headers = [ID_COLUMN, "L_joy", "L_tru", "L_fea", "L_sur", "L_sad", "L_ang"]
        pd.DataFrame(columns=headers).to_csv(OUTPUT_FILE, index=False)

    # 3. keep only rows still to do
    rows_to_process = df[~df[ID_COLUMN].astype(str).isin(processed_ids)]
    print(f"Posts left to score: {len(rows_to_process)} / {len(df)}")

    # 4. score and append line by line
    processed_count = 0
    with open(OUTPUT_FILE, "a") as f:
        for _, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process),
                           desc=f"Scoring {INPUT_FILE}"):
            post_id = str(row[ID_COLUMN])
            body = str(row[TEXT_COLUMN])[:1000]  # truncate very long posts to save tokens

            if pd.isna(body) or body.strip() == "":
                continue

            result = call_deepseek(client, prompt_level2(body))

            if result:
                f.write(
                    f'{post_id},{result.get("joy", 0)},{result.get("tru", 0)},'
                    f'{result.get("fea", 0)},{result.get("sur", 0)},'
                    f'{result.get("sad", 0)},{result.get("ang", 0)}\n'
                )
                f.flush()
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"\n[status] {processed_count} scored. Last ID: {post_id} -> {result}")
            else:
                print(f"\n[failed] Could not score ID {post_id} after {MAX_RETRIES} attempts.")

            time.sleep(SLEEP_SEC)

    print(f"\nDone. Results saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
