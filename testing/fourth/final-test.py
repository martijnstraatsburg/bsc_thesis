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
MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Consider trying more powerful models if available

class PostAnalyser:
    def __init__(self, client, model_name: str = MODEL, shot: str = "multi", temperature: float = 0.1, 
                 chain_of_thought: bool = True, use_structured_output: bool = True):
        """
        Initialize the post analyzer with improved parameters.
        
        Args:
            client: OpenAI client
            model_name: Model to use for analysis
            shot: "zero", "one", "few", "multi"
            temperature: Controls randomness, lower values make output more deterministic
            chain_of_thought: Whether to use chain-of-thought prompting
            use_structured_output: Whether to use structured JSON output format
        """
        self.client = client
        self.model = model_name
        self.shot = shot
        self.temperature = temperature
        self.chain_of_thought = chain_of_thought
        self.use_structured_output = use_structured_output
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create enhanced system prompt with persona and instructions."""
        base = """You are an expert linguist specializing in narrative analysis and text classification with years of experience in identifying stories in text.
        
You will classify ONE input passage at a time, in JSON, according to the following scheme.

1) STORY VS. NOT STORY
Use the following detailed criteria to determine if a text contains a story:

A story MUST have ALL of the following elements:
- A clear sequence of connected events (not just descriptions or opinions)
- One or more specific characters (people, animals, personified entities)
- Events that form a coherent, causally or temporally ordered chain
- Events that are singular occurrences in time ("he opened the door"; "she was astonished")

A text is NOT a story if it:
- Is primarily explanatory, argumentative, or informative without a narrative sequence
- Lacks specific characters or concrete events
- Is just a comment, reaction, or opinion without narrative elements
- Contains hypothetical scenarios without actual events that happened
- Is a general statement or observation without temporal progression

2) READERS' PERCEPTION RATINGS (1-5 Likert)
Based on Sternberg's narrative universals, rate:
  • SUSPENSE - "Triggers desire for information about future events, postpones a feeling of resolution."
  • CURIOSITY - "Triggers desire for information about past events, leaves me wondering about missing information."
  • SURPRISE - "Presents information that I experience as unexpected."
"""

        if self.chain_of_thought:
            base += """
You MUST follow this step-by-step reasoning process in your analysis:
1. Read the passage fully and carefully.
2. Identify any characters in the passage. List them specifically.
3. Identify any events in the passage. List them in sequence.
4. Check if the events form a coherent temporal or causal chain.
5. Based on your analysis of characters and events, determine if this is a story or not.
6. Provide your reasoning for the classification.
7. Rate Suspense (future information) 1-5. Ensure the rating is an integer between 1 and 5.
8. Rate Curiosity (past information) 1-5. Ensure the rating is an integer between 1 and 5.
9. Rate Surprise (unexpected information) 1-5. Ensure the rating is an integer between 1 and 5.
"""
        else:
            base += """
You MUST follow this process in your analysis:
1. Read the passage fully.
2. Decide whether the passage contains a story or not (Story vs. Not Story).
3. Rate Suspense (future information) 1-5. Ensure the rating is an integer between 1 and 5.
4. Rate Curiosity (past information) 1-5. Ensure the rating is an integer between 1 and 5.
5. Rate Surprise (unexpected information) 1-5. Ensure the rating is an integer between 1 and 5.
"""

        if self.use_structured_output:
            base += """
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
            ```

This is classified as a "Story" because:
1. It has a specific character (the narrator/author)
2. It describes specific events (attempting suicide, failing, receiving timely talks)
3. These events form a coherent sequence in time
4. The events are singular occurrences, not general descriptions""",
            "few": """```json
            [
                {
                    "body": "I attempted to kill myself, once. I failed.\n\nI have only told a few of my closest friends, and random people on the Internet who can't ID me.\n\nIncompetence is not insincerity. Were it not for a timely couple of talks from some amazing people, I would have gone ahead with my Mark II plan, which was substantially more lethal, and I would not be here.",
                    "story_class": "Story",
                    "suspense": 4,
                    "curiosity": 5,
                    "surprise": 4,
                    "explanation": "This is a story because it has a specific narrator, describes specific events (suicide attempt, conversations), and has a temporal sequence."
                },
                {
                    "body": "Sort of. The *general* phenomenon that describes the tendency for people to be attracted to similar-looking people doesn't have a specific term, it's just one factor among many that determines attraction. An informal naming of it would be \"like-attracts-like\", as Wikipedia describes it.\n\nA more specific idea within this idea does have a term: [Matching Hypothesis](http://en.wikipedia.org/wiki/Matching_hypothesis), which focuses on one's attraction to someone who is just as socially desirable as one's self.",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 2,
                    "explanation": "This is not a story because it contains no specific characters or events. It's an explanatory text about attraction patterns with no narrative sequence."
                },
                {
                    "body": "As someone who has parked illegally a few times, I was not aware that it was not legal to park there because I had always parked there at night. Once I found out, I moved my car.\n\nVandalism should never be \"okay\", especially since you can just call a tow truck and have it moved. If you want to mess up someone's car, it's unlikely you'll ever be caught. Just do it and stop trying to morally justify it.\n\ntl;dr: Don't always assume someone is just being a malicious dick, and don't retaliate.",
                    "story_class": "Story",
                    "suspense": 3,
                    "curiosity": 2,
                    "surprise": 4,
                    "explanation": "This is a story because it has a narrator as character, describes specific events (parking illegally, learning it was illegal, moving car), with a clear temporal sequence."
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
                    "surprise": 4,
                    "explanation": "This is a story because it has a specific narrator, describes specific events (suicide attempt, conversations), and has a temporal sequence."
                },
                {
                    "body": "Sort of. The *general* phenomenon that describes the tendency for people to be attracted to similar-looking people doesn't have a specific term, it's just one factor among many that determines attraction. An informal naming of it would be \"like-attracts-like\", as Wikipedia describes it.\n\nA more specific idea within this idea does have a term: [Matching Hypothesis](http://en.wikipedia.org/wiki/Matching_hypothesis), which focuses on one's attraction to someone who is just as socially desirable as one's self.",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 2,
                    "explanation": "This is not a story because it contains no specific characters or events. It's an explanatory text about attraction patterns with no narrative sequence."
                },
                {
                    "body": "As someone who has parked illegally a few times, I was not aware that it was not legal to park there because I had always parked there at night. Once I found out, I moved my car.\n\nVandalism should never be \"okay\", especially since you can just call a tow truck and have it moved. If you want to mess up someone's car, it's unlikely you'll ever be caught. Just do it and stop trying to morally justify it.\n\ntl;dr: Don't always assume someone is just being a malicious dick, and don't retaliate.",
                    "story_class": "Story",
                    "suspense": 3,
                    "curiosity": 2,
                    "surprise": 4,
                    "explanation": "This is a story because it has a narrator as character, describes specific events (parking illegally, learning it was illegal, moving car), with a clear temporal sequence."
                },
                {
                    "body": "What? He doesn't overreact in the sligh--POCKET SAND!",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 3,
                    "surprise": 5,
                    "explanation": "This is not a story because it lacks a sequence of connected events and doesn't establish a coherent narrative chain."
                },
                {
                    "body": "so he *did* kill himself",
                    "story_class": "Not Story",
                    "suspense": 1,
                    "curiosity": 3,
                    "surprise": 4,
                    "explanation": "This is not a story because it's a short comment or reaction without any narrative sequence of events."
                },
                {
                    "body": "Oh yeah, Zodd did beat him once.\n\nThat random commander too, but only because his sword broke.\n\nGood memory.",
                    "story_class": "Story",
                    "suspense": 1,
                    "curiosity": 2,
                    "surprise": 2,
                    "explanation": "This is a story because it mentions specific characters (Zodd, a commander) and describes specific events (beating, sword breaking) in a temporal sequence."
                },
                {
                    "body": "There is a surprisingly large number of people who actually believe that Courtney Love orchestrated his death. I can't really link anything cause I'm on mobile but the conspiracy theory is huge if you just google it. There's entire books and documentaries dedicated to it.",
                    "story_class": "Not Story",
                    "suspense": 2,
                    "curiosity": 4,
                    "surprise": 3,
                    "explanation": "This is not a story because it's an informative comment about a conspiracy theory without a narrative sequence of events."
                }
            ]
            ```"""
        }
        if self.shot == "zero":
            return base + "\n\nDo not include any text outside of this JSON format. Ensure your response is valid JSON. Ensure the ratings are integers (1-5)."
        else:
            explanation = "\n\nThe examples above show correct classifications. Pay special attention to the distinction between stories that have specific events in sequence versus non-stories that are just explanations, opinions, or comments without narrative elements."
            return (
                base
                + "\n\n" + "EXAMPLES:\n\n" +
                examples[self.shot] + explanation + "\n\n" +
                "Do not include any text outside of this JSON format. Ensure your response is valid JSON. Ensure the ratings are integers (1-5)."
            )

    def analyse_text(self, text: str) -> Dict[str, Any]:
        """Analyse the provided text and return structured results with improved error handling."""
        prompt = f"Text to analyse:\n\n{text}"

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,  # Using lower temperature for more consistent results
                response_format={"type": "json_object"} if self.use_structured_output else None
            )

            response_text = response.choices[0].message.content

            try:
                result = json.loads(response_text)
                # Validate the result to ensure it has the expected format
                if "story_class" not in result:
                    raise ValueError("Missing 'story_class' in response")
                if result["story_class"] not in ["Story", "Not Story"]:
                    result["story_class"] = "Not Story" if result["story_class"].lower() in ["not story", "not a story", "no story"] else "Story"
                
                # Ensure ratings are integers between 1-5
                for key in ["suspense", "curiosity", "surprise"]:
                    if key not in result:
                        result[key] = 1
                    else:
                        try:
                            result[key] = max(1, min(5, int(result[key])))
                        except (ValueError, TypeError):
                            result[key] = 1
                
                return result
            except json.JSONDecodeError:
                # Improved JSON extraction for potential formatting issues
                import re
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        # Apply same validation as above
                        if "story_class" not in result:
                            raise ValueError("Missing 'story_class' in response")
                        if result["story_class"] not in ["Story", "Not Story"]:
                            result["story_class"] = "Not Story" if result["story_class"].lower() in ["not story", "not a story", "no story"] else "Story"
                        
                        for key in ["suspense", "curiosity", "surprise"]:
                            if key not in result:
                                result[key] = 1
                            else:
                                try:
                                    result[key] = max(1, min(5, int(result[key])))
                                except (ValueError, TypeError):
                                    result[key] = 1
                        
                        return result
                    except:
                        pass
                
                # Fallback: attempt to extract classification from text
                story_match = re.search(r'story_class["\']\s*:\s*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
                story_class = "Not Story"  # Default
                if story_match:
                    extracted = story_match.group(1)
                    story_class = "Story" if "story" in extracted.lower() and "not" not in extracted.lower() else "Not Story"
                
                # Extract ratings if possible
                ratings = {}
                for key in ["suspense", "curiosity", "surprise"]:
                    rating_match = re.search(f'{key}["\']\s*:\s*(\d+)', response_text, re.IGNORECASE)
                    ratings[key] = int(rating_match.group(1)) if rating_match else 1
                    ratings[key] = max(1, min(5, ratings[key]))
                
                return {
                    "story_class": story_class,
                    "suspense": ratings["suspense"],
                    "curiosity": ratings["curiosity"],
                    "surprise": ratings["surprise"],
                    "extraction_method": "regex_fallback"
                }

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            # Default response in case of error
            return {
                "story_class": "Not Story",  # Default to the majority class
                "suspense": 1,
                "curiosity": 1,
                "surprise": 1,
                "error": str(e)
            }

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


def run_ensemble_classification(client, input_file: str, configurations: List[Dict], model_name: str = MODEL, output_dir: str = "results"):
    """Run ensemble classification using the specified configurations with voting."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)

    # Run each configuration
    config_results = {}
    
    for config in configurations:
        config_name = f"{config['shot']}_shot_temp{config['temperature']}_cot{'Yes' if config['chain_of_thought'] else 'No'}"
        print(f"\n--- Processing ensemble with {config_name} ---")
        
        # Create analyzer with the configuration
        analyzer = PostAnalyser(
            client=client, 
            model_name=model_name, 
            shot=config['shot'],
            temperature=config['temperature'],
            chain_of_thought=config['chain_of_thought']
        )
        
        config_results[config_name] = []
        
        # Process each entry
        for entry in tqdm(gold_standard, desc=f"Ensemble - {config_name}"):
            text = entry["body"]
            
            # Analyze the text
            analysis = analyzer.analyse_text(text)
            
            # Store result
            config_results[config_name].append({
                "id": entry["id"],
                "body": text,
                "analysis": analysis
            })
    
    # Create ensemble classifications
    ensemble_results = []
    for i, entry in enumerate(gold_standard):
        entry_id = entry["id"]
        body = entry["body"]
        
        # Collect all classifications
        classifications = []
        for config_name in config_results:
            classifications.append(config_results[config_name][i]["analysis"]["story_class"])
        
        # Determine majority vote
        story_count = classifications.count("Story")
        not_story_count = classifications.count("Not Story")
        
        majority_class = "Story" if story_count > not_story_count else "Not Story"
        
        # Calculate average ratings
        avg_suspense = sum(config_results[config_name][i]["analysis"].get("suspense", 1) 
                         for config_name in config_results) / len(config_results)
        
        avg_curiosity = sum(config_results[config_name][i]["analysis"].get("curiosity", 1)
                          for config_name in config_results) / len(config_results)
        
        avg_surprise = sum(config_results[config_name][i]["analysis"].get("surprise", 1)
                         for config_name in config_results) / len(config_results)
        
        # Create ensemble result
        ensemble_result = {
            "id": entry_id,
            "body": body,
            "story_class": majority_class,
            "suspense": round(avg_suspense),
            "curiosity": round(avg_curiosity),
            "surprise": round(avg_surprise)
        }
        
        ensemble_results.append(ensemble_result)
    
    # Save ensemble results
    output_file = os.path.join(output_dir, "ensemble_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ensemble_results, f, indent=2, ensure_ascii=False)
    print(f"Ensemble results saved to {output_file}")
    
    return ensemble_results


def process_gold_standard(client, input_file: str, model_name: str = MODEL, output_dir: str = "results"):
    """Process the gold standard dataset with improved methods."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)

    # Enhanced shot techniques with different parameters
    configurations = [
        {"shot": "zero", "temperature": 0.1, "chain_of_thought": False}#,
        #{"shot": "one", "temperature": 0.1, "chain_of_thought": False},
        #{"shot": "few", "temperature": 0.1, "chain_of_thought": True},
        #{"shot": "multi", "temperature": 0.1, "chain_of_thought": True},
        # Additional configurations for better coverage
        #{"shot": "multi", "temperature": 0.2, "chain_of_thought": True},
        #{"shot": "multi", "temperature": 0.1, "chain_of_thought": False} 
    ]

    for config in configurations:
        config_name = f"{config['shot']}_shot_temp{config['temperature']}_cot{'Yes' if config['chain_of_thought'] else 'No'}"
        print(f"\n--- Processing with {config_name} ---")

        analyser = PostAnalyser(
            client=client, 
            model_name=model_name, 
            shot=config['shot'],
            temperature=config['temperature'],
            chain_of_thought=config['chain_of_thought']
        )

        results = copy.deepcopy(gold_standard)

        # Process each entry
        for i, entry in enumerate(tqdm(results, desc=config_name)):
            text = entry["body"]
            analysis = analyser.analyse_text(text)

            if "error" not in analysis:
                entry["story_class"] = analysis["story_class"]
                entry["suspense"] = analysis["suspense"]
                entry["curiosity"] = analysis["curiosity"]
                entry["surprise"] = analysis["surprise"]
            else:
                print(f"Error analyzing entry {i+1}: {analysis['error']}")
                entry["story_class"] = "Not Story"  # Default to majority class as fallback
                entry["suspense"] = 1
                entry["curiosity"] = 1
                entry["surprise"] = 1

        output_file = os.path.join(output_dir, f"gold_standard_{config_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results for {config_name} saved to {output_file}")

    # Run ensemble classification with the same configurations
    run_ensemble_classification(client, input_file, configurations, model_name, output_dir)


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


def evaluate_results(gold_standard_file: str, results_dir: str, output_file: str = "improved_evaluation_metrics.json"):
    with open(gold_standard_file, 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)
    id_to_label = {entry['id']: entry['story_class'] for entry in gold_standard}
    
    # Get all result files including new ones
    result_files = [f for f in os.listdir(results_dir) if f.startswith("gold_standard_") and f.endswith(".json")]
    # Also check for ensemble results
    if os.path.exists(os.path.join(results_dir, "ensemble_results.json")):
        result_files.append("ensemble_results.json")

    eval_summary = {}
    for result_file in result_files:
        if result_file == "ensemble_results.json":
            config_name = "ensemble"
        else:
            config_name = result_file.replace("gold_standard_", "").replace(".json", "")
        
        with open(os.path.join(results_dir, result_file), 'r', encoding='utf-8') as f:
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

        eval_summary[config_name] = {
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
    
    # Find and display the best configuration
    best_accuracy = 0
    best_config = None
    for config, metrics in eval_summary.items():
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_config = config
    
    print(f"\nBest configuration: {best_config} with accuracy: {best_accuracy:.4f}")
    return eval_summary


if __name__ == "__main__":
    client = OpenAI(base_url=BASEURL, api_key=APIKEY)

    mod_gold_standard_file = "mod-gs-train.json"
    gold_standard_file = "gs-train.json"
    output_directory = "improved_llm_analysis_results"

    # Define the configurations you want to use - this is the main change
    configurations = [
        {"shot": "zero", "temperature": 0.1, "chain_of_thought": False}#,
        #{"shot": "one", "temperature": 0.1, "chain_of_thought": False},
        #{"shot": "few", "temperature": 0.1, "chain_of_thought": True},
        #{"shot": "multi", "temperature": 0.1, "chain_of_thought": True},
        #{"shot": "multi", "temperature": 0.2, "chain_of_thought": True},
        #{"shot": "multi", "temperature": 0.1, "chain_of_thought": False} 
    ]

    # Process individual configurations
    for config in configurations:
        config_name = f"{config['shot']}_shot_temp{config['temperature']}_cot{'Yes' if config['chain_of_thought'] else 'No'}"
        print(f"\n--- Processing with {config_name} ---")

        analyser = PostAnalyser(
            client=client, 
            model_name=MODEL, 
            shot=config['shot'],
            temperature=config['temperature'],
            chain_of_thought=config['chain_of_thought']
        )

        with open(mod_gold_standard_file, 'r', encoding='utf-8') as f:
            gold_standard = json.load(f)

        results = copy.deepcopy(gold_standard)

        # Process each entry
        for i, entry in enumerate(tqdm(results, desc=config_name)):
            text = entry["body"]
            analysis = analyser.analyse_text(text)

            if "error" not in analysis:
                entry["story_class"] = analysis["story_class"]
                entry["suspense"] = analysis["suspense"]
                entry["curiosity"] = analysis["curiosity"]
                entry["surprise"] = analysis["surprise"]
            else:
                print(f"Error analyzing entry {i+1}: {analysis['error']}")
                entry["story_class"] = "Not Story"  # Default to majority class as fallback
                entry["suspense"] = 1
                entry["curiosity"] = 1
                entry["surprise"] = 1

        output_file = os.path.join(output_directory, f"gold_standard_{config_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results for {config_name} saved to {output_file}")

    # Run ensemble classification directly with our defined configurations
    run_ensemble_classification(client, mod_gold_standard_file, configurations, MODEL, output_directory)
    
    # Evaluate all configurations
    evaluation = evaluate_results(gold_standard_file, output_directory, "improved_evaluation_metrics.json")
    
    # Test the best configuration
    best_config = max(evaluation.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nUsing best configuration: {best_config}")
    