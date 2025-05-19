#!/usr/bin/env python3

# Name: qwen3-prompting.py
# Author: Martijn Straatsburg
# Description: This script prompts either of the Qwen3 models (4B or 8B) with a set of configurations.
# The configs can be adjusted to have different shot-techniques, temperatures, yes/no Chain-of-Thought, yes/no structured output.
# Basically the same script as prompt-cv-eval.py, but only prompting for the predictions.

import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import json
from typing import Dict, Any, List, Union
import copy

BASEURL = "http://localhost:8000/v1/"
APIKEY = "EMPTY"
MODEL = "Qwen/Qwen3-4B"


class PostAnalyser:
    def __init__(self, client, model_name: str = MODEL, shot: str = "multi", temperature: float = 0.6, 
                 chain_of_thought: bool = True, use_structured_output: bool = True):
        """
        ...
        """
        self.client = client
        self.model = model_name
        self.shot = shot
        self.temperature = temperature
        self.chain_of_thought = chain_of_thought
        self.use_structured_output = use_structured_output
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt with persona and instructions.
        Returns:
            str: The system prompt for the model.
        """
        base = """You are an empathic, caring, world-curious, eager linguist specializing in narrative analysis and text classification.
        
You will classify ONE input text at a time, in JSON, according to the following guidelines.
Use the guidelines below to support your decisions, but ultimately, follow your best judgment and more than not it is a story.
A text is a story if it describes a sequence of events involving one or more people. Stories must include multiple events.
It's ok if the events are out of order, but there should still be some sequence.
These texts can contain multiple stories, or one small story, or no story at all. Most of the time, the text will contain a story.
So, if you are not sure, classify it as a story.
Stories can be experienced in different ways, and the same story can be experienced differently by different readers.
These experiences are called readers' perceptions.
There are three readers' perceptions you will rate within these texts:\n
- SUSPENSE: "Triggers desire for information about future events and postpones a feeling of resolution."
- CURIOSITY: "Triggers desire for information about past events and leaves me wondering about missing information."
- SURPRISE: "Presents information that I experience as unexpected."

Task:
Decide whether the text CONTAINS a story or not AND rate your readers' perceptions of the text on a scale from 1 to 5.
\n
\n
"""
        if self.chain_of_thought:
            base += """
Let's think step by step. Follow these steps to complete the task:\n
1. Read the text fully.
2. Decide whether the text CONTAINS a story or not. Most of the time, the text will contain a story you might have missed.
3. Rate SUSPENSE (future information) on a scale from 1 to 5.
4. Rate CURIOSITY (past information) on a scale from 1 to 5.
5. Rate SURPRISE (unexpected information) on a scale from 1 to 5.
6. Provide the final output in JSON format. Ensure the output is valid JSON. Ensure the ratings are integers between 1 and 5.
\n
\n
"""
        if self.use_structured_output:
            base += """
You MUST provide the output in a structured JSON format with the following structure:\n
{
  "story_class": "Story" or "Not Story",
  "suspense": integer (1-5),
  "curiosity": integer (1-5),
  "surprise": integer (1-5)
}
\n
"""
        examples = {
            "one": """```json
            [
                {
                    "body": "As I read it, the OP did not consent even when drunk. Those the consent is dubious not just because of the alcohol involved but also because I think there might not have been any consent whatsoever. \n\nFurthermore, on Mandy's part, if she's partying with the secondary intention of banging the OP even after he's made it completely clear while relatively sober that he does not want to have sex with her, she is entirely to blame. The OP's lack of consent was not just implied (through an absence of active consent) but actually explicit. Now it's possible that Mandy was drinking with no intention of banging the OP but that after enough alcohol, she couldn't control herself even though she expected she would be able to. Then perhaps she's not entirely to blame, but any residual blame goes to the alcohol rather than to the OP. \n\nNow, if you were to wake up in someone's bed after you had told that person explicitly while sober, \"I don't want to have sex with you,\" would you still be blaming yourself? Especially given that the other person initiated the act and you were too out of it to really do anything to prevent it?",
    "story_class": "Story",
    "suspense": 4,
    "curiosity": 3,
    "surprise": 4
                }
            ]
            ```""",
            "few": """```json
            [
                {
                    "body": "As I read it, the OP did not consent even when drunk. Those the consent is dubious not just because of the alcohol involved but also because I think there might not have been any consent whatsoever. \n\nFurthermore, on Mandy's part, if she's partying with the secondary intention of banging the OP even after he's made it completely clear while relatively sober that he does not want to have sex with her, she is entirely to blame. The OP's lack of consent was not just implied (through an absence of active consent) but actually explicit. Now it's possible that Mandy was drinking with no intention of banging the OP but that after enough alcohol, she couldn't control herself even though she expected she would be able to. Then perhaps she's not entirely to blame, but any residual blame goes to the alcohol rather than to the OP. \n\nNow, if you were to wake up in someone's bed after you had told that person explicitly while sober, \"I don't want to have sex with you,\" would you still be blaming yourself? Especially given that the other person initiated the act and you were too out of it to really do anything to prevent it?",
    "story_class": "Story",
    "suspense": 4,
    "curiosity": 3,
    "surprise": 4
                },
                {
                    "body": "Rooney extremely overreacted. \n\nIt's one thing to pursue a student for being absent, it's another thing to devote his entire day to it and attempt to break into his home.",
    "story_class": "Not Story",
    "suspense": 3,
    "curiosity": 2,
    "surprise": 3
                },
                {
                    "body": "Students and cafeteria loos, usually disaster. I checked a dozen loo berths and it was all underhang. I even took a picture in disbelief! \n\nNow I think it must have been policy, sick of finding rolls \"spun\" onto the floor.",
    "story_class": "Story",
    "suspense": 1,
    "curiosity": 2,
    "surprise": 2
                }
            ]
            ```""",
            "multi": """```json
            [
                {
                    "body": "As I read it, the OP did not consent even when drunk. Those the consent is dubious not just because of the alcohol involved but also because I think there might not have been any consent whatsoever. \n\nFurthermore, on Mandy's part, if she's partying with the secondary intention of banging the OP even after he's made it completely clear while relatively sober that he does not want to have sex with her, she is entirely to blame. The OP's lack of consent was not just implied (through an absence of active consent) but actually explicit. Now it's possible that Mandy was drinking with no intention of banging the OP but that after enough alcohol, she couldn't control herself even though she expected she would be able to. Then perhaps she's not entirely to blame, but any residual blame goes to the alcohol rather than to the OP. \n\nNow, if you were to wake up in someone's bed after you had told that person explicitly while sober, \"I don't want to have sex with you,\" would you still be blaming yourself? Especially given that the other person initiated the act and you were too out of it to really do anything to prevent it?",
    "story_class": "Story",
    "suspense": 4,
    "curiosity": 3,
    "surprise": 4
                },
                {
                    "body": "Rooney extremely overreacted. \n\nIt's one thing to pursue a student for being absent, it's another thing to devote his entire day to it and attempt to break into his home.",
    "story_class": "Not Story",
    "suspense": 3,
    "curiosity": 2,
    "surprise": 3
                },
                {
                    "body": "Students and cafeteria loos, usually disaster. I checked a dozen loo berths and it was all underhang. I even took a picture in disbelief! \n\nNow I think it must have been policy, sick of finding rolls \"spun\" onto the floor.",
    "story_class": "Story",
    "suspense": 1,
    "curiosity": 2,
    "surprise": 2
                },
                {
                    "body": "The physical evidence shows that he was not turned around and that he was moving toward the police officer. \n\nYou can't argue with physical evidence.",
    "story_class": "Not Story",
    "suspense": 2,
    "curiosity": 2,
    "surprise": 3
                },
                {
                    "body": "About a month ago I posted a CMV about how I think tipping should be made illegal (my view was NOT changed), and today I read [this](http://jayporter.com/dispatches/observations-from-a-tipless-restaurant-part-1-overview/) article about a San Diego restaurant that forbids tips!",
    "story_class": "Story",
    "suspense": 3,
    "curiosity": 2,
    "surprise": 2
                },
                {
                    "body": "Nobody is calling the police questioning him \"police brutality.\"  It's all the physical intimidation and pushing him on the ground.",
    "story_class": "Not Story",
    "suspense": 4,
    "curiosity": 2,
    "surprise": 2
                },
                {
                    "body": "It is not necessary to have a complex character for a character study in feature film. The character only needs to be interesting. I think that Plainview is plenty interesting. There was some ambiguity in his character that keeps you intrigued. There is obviously something to his relationship with H.W. that goes beyond simple exploitation. Saving him from the oil well and the way he cradle him afterwards shows that there is some sort feelings there. Also how he greets H.W. on his return from the school again shows that something has developed between them. Also his interactions with Henry show that he wanted some sort of human relationship otherwise he would of just sent him away after their first meeting.\n\nI didn't have any problem with the structure of the movie or the leaps in time. This is the first I've heard of anybody having issue with that as well. I don't know what else to say about that.\n\nI don't think broader context was really necessary for the film and in fact think that the intense focus on Plainview was one of its strengths. I think broadening the focus would have meant that the film would have lost its tension.\n\nI don't see Plainview killing Eli as puzzling. In fact I think it is really the only ending that would have made sense. What happens to somebody that is driven by success to sociopathic ends after they win that success? They turn on themselves and anybody around them. This is why it also showed Plainview pushing H.W. out of his life for good.",
    "story_class": "Story",
    "suspense": 2,
    "curiosity": 4,
    "surprise": 3
                },
                {
                   "body": "He also had devil traps under all his rugs, a car full of shotguns, saw monsters everywhere and almost killed his neighbor's dog.\n\nComparatively, Sam just had a house, a dog, a bitch, and no crazy.",
    "story_class": "Story",
    "suspense": 1,
    "curiosity": 3,
    "surprise": 4
                }
            ]
            ```"""
        }
        if self.shot == "zero":
            return base + "\n\nDo not include any text outside of this JSON format. Ensure your response is valid JSON."
        else:
            return (
                base
                + "\n\n" + "EXAMPLES:\n\n" +
                examples[self.shot] + "\n\n" +
                "Do not include any text outside of this JSON format. Ensure your response is valid JSON."
            )

    def analyse_text(self, text: str) -> Dict[str, Any]:
        """
        ...
        """
        prompt = f"Text to analyse:\n\n{text}"
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"} if self.use_structured_output else None
            )
            response_text = response.choices[0].message.content
            try:
                result = json.loads(response_text)
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
            except json.JSONDecodeError:
                import re
                json_match = re.search(r"({.*})", response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
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
                story_match = re.search(r'story_class["\']\s*:\s*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
                story_class = "Not Story"
                if story_match:
                    extracted = story_match.group(1)
                    story_class = "Story" if "story" in extracted.lower() and "not" not in extracted.lower() else "Not Story"
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
            return {
                "story_class": "Not Story",
                "suspense": 1,
                "curiosity": 1,
                "surprise": 1,
                "error": str(e)
            }

    def batch_analyse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        ...
        """
        results = []
        for text in tqdm(texts, desc="Analyzing texts"):
            result = self.analyse_text(text)
            results.append(result)
        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """
        ...
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")


def process_gold_standard(client, input_file: str, model_name: str = MODEL, output_dir: str = "results"):
    """
    ...
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f:
        gold_standard = json.load(f)
    configurations = [
        {"shot": "few", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True}
    ]
    for config in configurations:
        config_name = f"{config['shot']}_shot_temp{config['temperature']}_cot{'Yes' if config['chain_of_thought'] else 'No'}_structured{'Yes' if config['use_structured_output'] else 'No'}"
        print(f"\n--- Processing with {config_name} ---")
        analyser = PostAnalyser(
            client=client, 
            model_name=model_name, 
            shot=config['shot'],
            temperature=config['temperature'],
            chain_of_thought=config['chain_of_thought'],
            use_structured_output=config['use_structured_output']
        )
        results = copy.deepcopy(gold_standard)
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
                entry["story_class"] = "Not Story"
                entry["suspense"] = 1
                entry["curiosity"] = 1
                entry["surprise"] = 1
        output_file = os.path.join(output_dir, f"gold_standard_{config_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results for {config_name} saved to {output_file}")


if __name__ == "__main__":
    client = OpenAI(base_url=BASEURL, api_key=APIKEY)
    mod_gold_standard_file = "part_1.json"
    output_directory = "prediction-phase" # Change filename after or it overwrites
    process_gold_standard(client, mod_gold_standard_file, model_name=MODEL, output_dir=output_directory)
    analyser = PostAnalyser(client=client, model_name=MODEL, shot="multi", temperature=0.6,
                            chain_of_thought=True, use_structured_output=True)
    test_text = "I believe that the conclusions reached by the House Select Committee on Assassinations (HSCA) regarding the assassination of JFK on Nov. 22, 1963 are true and accurate. CMV.\n\nI believe that Lee Harvey Oswald assassinated JFK using a single rifle, fired all of the shots that killed him and wounded Gov. Connally, and did so as an individual actor (not as the agent of a nation or organized group). My knowledge of the Kennedy Assassination is admittedly limited; I've studied it in school and participated in several university level class discussions about the investigations and conspiracies surrounding the issue. After looking at the conclusions reached by the Warren Commission, I was somewhat skeptical of the notion that Lee Harvey Oswald acted alone, but the HSCA report clarifies some of the issues surrounding the timing and circumstances of the assassination, and leaves open the idea that Oswald may have had a partner. Additionally, I do feel that the events following the assassination are slightly suspicious (notably LBJ appointing and overseeing the Warren Commission), but are not at present sufficient for me to discount the accepted explanation of the assassination.  The number of conspiracy or alternate theories surrounding the assassination is prodigious and confusing, to say the least, but I am willing to accept an alternate theory and CMV if someone can present a compelling alternate explanation of the JFK assassination supported by clear and provable facts."
    result = analyser.analyse_text(test_text)
    print("\nTest analysis result:")
    print(json.dumps(result, indent=2))
