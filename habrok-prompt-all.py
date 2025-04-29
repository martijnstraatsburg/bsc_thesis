import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import json
from typing import Dict, Any, List, Union
import copy

BASEURL = 'http://localhost:8000/v1/'
APIKEY = 'EMPTY'
MODEL = "Qwen/Qwen2.5-7B-Instruct"

class PostAnalyser:
    def __init__(self, client, model_name: str = MODEL, shot: str = "zero"):
        """
        shot: "zero", "one", "few", "multi"
        """
        self.client = client
        self.model = model_name
        self.shot = shot
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt with persona and instructions."""
        base = """You are an expert linguist specializing in narrative analysis and text classification.
        
You will classify ONE input passage at a time, in JSON, according to the following scheme.
1) STORY VS. NOT STORY
Use the Codebook's constructivist definition of a story:
- A story describes a sequence of connected events involving one or more specific characters (people, animals, etc.).
- Events are singular occurrences in time ("he opened the door"; "she was astonished"), and must form a coherent, causally or temporally ordered chain.

2) READERS' PERCEPTION RATINGS (1-5 Likert)
Based on Sternberg's narrative universals, rate:
  • SUSPENSE - "Triggers desire for information about future events, postpones a feeling of resolution."
  • CURIOSITY - "Triggers desire for information about past events, leaves me wondering about missing information."
  • SURPRISE - "Presents information that I experience as unexpected."

You MUST follow this process in your analysis:
1. Read the passage fully.
2. Decide whether the passage contains a story or not (Story vs. Not Story (use the Codebook)).
3. Rate Suspense (future information) 1-5. Ensure the rating is an integer between 1 and 5.
4. Rate Curiosity (past information) 1-5. Ensure the rating is an integer between 1 and 5.
5. Rate Surprise (unexpected information) 1-5. Ensure the rating is an integer between 1 and 5.

Your response MUST be valid JSON with the following structure:
{
  "story_class": "Story" or "Not Story",
  "suspense": integer (1-5),
  "curiosity": integer (1-5),
  "surprise": integer (1-5)
}
"""
        examples = {
            "one": """```json
            [
                {
                    "body": "...",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 3,
                    "surprise": 5
                }
            ]
            ```""",
            "few": """```json
            [
                {
                    "body": "...",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 3,
                    "surprise": 5
                },
                {
                    "body": "...",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 1
                },
                {
                    "body": "...",
                    "story_class": "Story",
                    "suspense": 5,
                    "curiosity": 4,
                    "surprise": 3
                }
            ]
            ```""",
            "multi": """```json
            [
                {
                    "body": "...",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 3,
                    "surprise": 5
                },
                {
                    "body": "...",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 1
                },
                {
                    "body": "...",
                    "story_class": "Story",
                    "suspense": 5,
                    "curiosity": 4,
                    "surprise": 3
                },
                {
                    "body": "...",
                    "story_class": "Not Story",
                    "suspense": 2,
                    "curiosity": 1,
                    "surprise": 2
                },
                {
                    "body": "...",
                    "story_class": "Story",
                    "suspense": 3,
                    "curiosity": 5,
                    "surprise": 4
                },
                {
                    "body": "...",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 1
                },
                {
                    "body": "...",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 3,
                    "surprise": 5
                },
                {
                    "body": "...",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 1
                }
            ]
            ```"""
        }
        if self.shot == "zero":
            return base + "\n\nDo not include any text outside of this JSON format. Ensure your response is valid JSON. Ensure the ratings are integers (1-5)."
        else:
            return (
                base
                + "\n\n" + "EXAMPLES:\n\n" +
                examples[self.shot] + "\n\n" +
                "Do not include any text outside of this JSON format. Ensure your response is valid JSON. Ensure the ratings are integers (1-5)."
            )

    def analyse_text(self, text: str) -> Dict[str, Any]:
        """Analyse the provided text and return structured results."""
        prompt = f"Text to analyse:\n\n{text}"

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content

            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                    return result
                else:
                    return {
                        "error": "Failed to parse JSON response",
                        "raw_response": response_text
                    }

        except Exception as e:
            return {"error": str(e)}

    def batch_analyse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyse multiple texts and return results for each."""
        results = []
        for text in tqdm(texts, desc="Analyzing texts"):
            result = self.analyse_text(text)
            results.append(result)
        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """Save analysis results to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")


def process_gold_standard(client, input_file: str, model_name: str = MODEL, output_dir: str = "results"):
    """Process the gold standard dataset with different 'shot' techniques."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)

    shot_techniques = ["zero", "one", "few", "multi"]

    for shot in shot_techniques:
        print(f"\n--- Processing with {shot}-shot technique ---")

        analyser = PostAnalyser(client=client, model_name=model_name, shot=shot)

        results = copy.deepcopy(gold_standard)

        # Process each entry
        for i, entry in enumerate(tqdm(results, desc=f"{shot}-shot")):
            if all(key in entry for key in ["story_class", "suspense", "curiosity", "surprise"]):
                print(f"Skipping entry {i+1} as it already has classification data")
                continue

            text = entry["body"]

            analysis = analyser.analyse_text(text)

            if "error" not in analysis:
                entry["story_class"] = analysis["story_class"]
                entry["suspense"] = analysis["suspense"]
                entry["curiosity"] = analysis["curiosity"]
                entry["surprise"] = analysis["surprise"]
            else:
                print(f"Error analyzing entry {i+1}: {analysis['error']}")
                entry["story_class"] = "Error"
                entry["suspense"] = 0
                entry["curiosity"] = 0
                entry["surprise"] = 0

        output_file = os.path.join(output_dir, f"gold_standard_{shot}_shot.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results for {shot}-shot saved to {output_file}")


def compare_results(gold_standard_file: str, results_dir: str, output_file: str = "comparison.json"):
    """Compare the analysis results with the gold standard."""
    with open(gold_standard_file, 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)
    result_files = [f for f in os.listdir(results_dir) if f.startswith("gold_standard_") and f.endswith(".json")]
    comparison = {}
    for entry in gold_standard:
        entry_id = entry["id"]
        comparison[entry_id] = {
            "gold_standard": {
                "story_class": entry["story_class"],
                "suspense": entry["suspense"],
                "curiosity": entry["curiosity"],
                "surprise": entry["surprise"]
            },
            "results": {}
        }

    for result_file in result_files:
        shot = result_file.replace("gold_standard_", "").replace("_shot.json", "")
        
        with open(os.path.join(results_dir, result_file), 'r', encoding='utf-8') as f:
            results = json.load(f)

        for entry in results:
            entry_id = entry["id"]
            if entry_id in comparison:
                comparison[entry_id]["results"][shot] = {
                    "story_class": entry["story_class"],
                    "suspense": entry["suspense"],
                    "curiosity": entry["curiosity"],
                    "surprise": entry["surprise"]
                }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Comparison results saved to {output_file}")


if __name__ == "__main__":
    client = OpenAI(base_url=BASEURL, api_key=APIKEY)

    mod_gold_standard_file = "modified_gold_standard.json"
    gold_standard_file = "small_gold_standard.json"
    output_directory = "llm_analysis_results"

    process_gold_standard(client, mod_gold_standard_file, model_name=MODEL, output_dir=output_directory)
    
    compare_results(gold_standard_file, output_directory)
    
    analyser = PostAnalyser(client=client, model_name=MODEL, shot="zero")
    test_text = "A man walked into a bar. He ordered a drink and then left."
    result = analyser.analyse_text(test_text)
    print("\nTest analysis result:")
    print(json.dumps(result, indent=2))
