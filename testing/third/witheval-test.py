import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import json
from typing import Dict, Any, List, Union
import copy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report


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
                    "body": "I attempted to kill myself, once. I failed.\n\nI have only told a few of my closest friends, and random people on the Internet who can't ID me.\n\nIncompetence is not insincerity. Were it not for a timely couple of talks from some amazing people, I would have gone ahead with my Mark II plan, which was substantially more lethal, and I would not be here.",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 5,
                    "surprise": 4
                }
            ]
            ```""",
            "few": """```json
            [
                {
                    "body": "I attempted to kill myself, once. I failed.\n\nI have only told a few of my closest friends, and random people on the Internet who can't ID me.\n\nIncompetence is not insincerity. Were it not for a timely couple of talks from some amazing people, I would have gone ahead with my Mark II plan, which was substantially more lethal, and I would not be here.",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 5,
                    "surprise": 4
                },
                {
                    "body": "Sort of. The *general* phenomenon that describes the tendency for people to be attracted to similar-looking people doesn't have a specific term, it's just one factor among many that determines attraction. An informal naming of it would be \"like-attracts-like\", as Wikipedia describes it.\n\nA more specific idea within this idea does have a term: [Matching Hypothesis](http://en.wikipedia.org/wiki/Matching_hypothesis), which focuses on one's attraction to someone who is just as socially desirable as one's self.",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 2
                },
                {
                    "body": "As someone who has parked illegally a few times, I was not aware that it was not legal to park there because I had always parked there at night. Once I found out, I moved my car.\n\nVandalism should never be \"okay\", especially since you can just call a tow truck and have it moved. If you want to mess up someone's car, it's unlikely you'll ever be caught. Just do it and stop trying to morally justify it.\n\ntl;dr: Don't always assume someone is just being a malicious dick, and don't retaliate.",
                    "story_class": "Story",
                    "suspense": 3,
                    "curiosity": 2,
                    "surprise": 4
                }
            ]
            ```""",
            "multi": """```json
            [
                {
                    "body": "I attempted to kill myself, once. I failed.\n\nI have only told a few of my closest friends, and random people on the Internet who can't ID me.\n\nIncompetence is not insincerity. Were it not for a timely couple of talks from some amazing people, I would have gone ahead with my Mark II plan, which was substantially more lethal, and I would not be here.",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 5,
                    "surprise": 4
                },
                {
                    "body": "Sort of. The *general* phenomenon that describes the tendency for people to be attracted to similar-looking people doesn't have a specific term, it's just one factor among many that determines attraction. An informal naming of it would be \"like-attracts-like\", as Wikipedia describes it.\n\nA more specific idea within this idea does have a term: [Matching Hypothesis](http://en.wikipedia.org/wiki/Matching_hypothesis), which focuses on one's attraction to someone who is just as socially desirable as one's self.",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 2
                },
                {
                    "body": "As someone who has parked illegally a few times, I was not aware that it was not legal to park there because I had always parked there at night. Once I found out, I moved my car.\n\nVandalism should never be \"okay\", especially since you can just call a tow truck and have it moved. If you want to mess up someone's car, it's unlikely you'll ever be caught. Just do it and stop trying to morally justify it.\n\ntl;dr: Don't always assume someone is just being a malicious dick, and don't retaliate.",
                    "story_class": "Story",
                    "suspense": 3,
                    "curiosity": 2,
                    "surprise": 4
                },
                {
                    "body": "What? He doesn't overreact in the sligh--POCKET SAND!",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 3,
                    "surprise": 5
                },
                {
                    "body": "Zunes were never cool. CMV.\n\nLet's face it: Zune (the media delivery service, the actual device) was just a blatant attempt by Microsoft to compete and stay relevant when faced with an ascendant Apple. I'm by no means an Apple fanboy, but nothing is less cool than copying wholesale what is currently popular. The mp3 players looked bad, the sites were godawful, and Zune is generally considered to be emblematic of the tack the Microsoft has taken to compete with Apple, which hasn't been great. \"Leasing\" songs to your Zune-having friends was two steps forward idea-wise, but 3 steps back CRM-wise. Watching the build up and \"hype\" to the release of Zune was like watching a car crash in slow motion.\n\nI'm not saying that Zune was useless or uninteresting or ineffective, my view is simply that it isn't, and never was, cool. Hip. Desirable to have. You will have changed my view when you make me (or some past version of me) desire a Zune.",
                    "story_class": "Story",
                    "suspense": 2,
                    "curiosity": 4,
                    "surprise": 2
                },
                {
                    "body": "so he *did* kill himself",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 3,
                    "surprise": 4
                },
                {
                    "body": "Oh yeah, Zodd did beat him once.\n\nThat random commander too, but only because his sword broke.\n\nGood memory.",
                    "story_class": "Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 2
                },
                {
                    "body": "There is a surprisingly large number of people who actually believe that Courtney Love orchestrated his death. I can't really link anything cause I'm on mobile but the conspiracy theory is huge if you just google it. There's entire books and documentaries dedicated to it.",
                    "story_class": "Not Story",
                    "suspense": 2,
                    "curiosity": 4,
                    "surprise": 3
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

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

def evaluate_results(gold_standard_file: str, results_dir: str, output_file: str = "evaluation_metrics.json"):
    with open(gold_standard_file, 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)
    id_to_label = {entry['id']: entry['story_class'] for entry in gold_standard}
    shot_files = [f for f in os.listdir(results_dir) if f.startswith("gold_standard_") and f.endswith(".json")]

    eval_summary = {}
    for shot_file in shot_files:
        shot = shot_file.replace("gold_standard_", "").replace("_shot.json", "")
        with open(os.path.join(results_dir, shot_file), 'r', encoding='utf-8') as f:
            results = json.load(f)
        y_true, y_pred = [], []
        for entry in results:
            eid = entry['id']
            if eid in id_to_label:
                y_true.append(id_to_label[eid])
                y_pred.append(entry.get('story_class', 'Error'))
        labels = sorted(list(set(y_true)))
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

        eval_summary[shot] = {
            'accuracy': acc,
            'precision': {lbl: precision[idx] for idx, lbl in enumerate(labels)},
            'recall': {lbl: recall[idx] for idx, lbl in enumerate(labels)},
            'f1_score': {lbl: f1[idx] for idx, lbl in enumerate(labels)},
            'support': {lbl: support[idx] for idx, lbl in enumerate(labels)},
            'classification_report': report
        }

    # Convert to JSON-serializable types
    eval_summary_clean = convert_numpy(eval_summary)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_summary_clean, f, indent=2, ensure_ascii=False)
    print(f"Evaluation metrics saved to {output_file}")


if __name__ == "__main__":
    client = OpenAI(base_url=BASEURL, api_key=APIKEY)

    mod_gold_standard_file = "mod-gs-train.json"
    gold_standard_file = "gs-train.json"
    output_directory = "llm_analysis_results"

    process_gold_standard(client, mod_gold_standard_file, model_name=MODEL, output_dir=output_directory)
    
    compare_results(gold_standard_file, output_directory)
    
    evaluate_results(gold_standard_file, output_directory)
    
    analyser = PostAnalyser(client=client, model_name=MODEL, shot="zero")
    test_text = "A man walked into a bar. He ordered a drink and then left."
    result = analyser.analyse_text(test_text)
    print("\nTest analysis result:")
    print(json.dumps(result, indent=2))
