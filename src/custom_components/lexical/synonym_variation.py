from collections import defaultdict
import logging
import random
from typing import List, Optional, Dict, Any, Tuple

from haystack import component, default_to_dict
from haystack.dataclasses import ChatMessage
import spacy
from lemminflect import getInflection
import wn

from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator

# Constants for POS mapping and supported POS tags
POS_MAPPING: Dict[str, str] = {
    "ADJ": "a",
    "ADV": "r",
    "NOUN": "n",
    "VERB": "v",
}

@component
class LexicalVariator:
    """
    A component that varies the lexical content of text by replacing words with their synonyms.
    
    This component uses WordNet to find synonyms for each word in the text and randomly samples one of them.
    It employs word sense disambiguation to select the most appropriate synset based on context.
    """
    
    def __init__(
        self,
        wsd_model: CachedOpenAIChatGenerator,
        spacy_model: str = "en_core_web_sm",
        wordnet_version: str = "oewn:2024",
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the LexicalVariator component.

        :param wsd_model: The model to use for word sense disambiguation. 
                         Should be a deployed version of swap-uniba/LLM-wsd-FT-ALL through vLLM.
        :param spacy_model: The spaCy model to use for text analysis.
        :param wordnet_version: The WordNet version to use for synonym lookup.
        :param random_seed: Optional random seed for reproducibility.
        """
        try:
            self.spacy_model = spacy.load(spacy_model)
        except OSError:
            spacy.cli.download(spacy_model)
            self.spacy_model = spacy.load(spacy_model)
        
        try:
            self.wordnet = wn.Wordnet(wordnet_version)
        except wn.Error:
            wn.download(wordnet_version)
            self.wordnet = wn.Wordnet(wordnet_version)

        self.wsd_model = wsd_model
        self.spacy_model_name = spacy_model
        self.wordnet_version = wordnet_version
        self.random_seed = random_seed
        random.seed(random_seed)

    @component.output_types(text=str, metadata=Dict[str, Any])
    def run(self, text: str, context: Optional[str] = None) -> Dict[str, str]:
        """
        Vary the lexical content of a text by replacing words with their synonyms.

        :param text: The text to vary.
        :param context: Context to use for word sense disambiguation.
        """
        # Apply spaCy to the text for POS tagging and lemmatization
        analyzed_text = self.spacy_model(text)

        if context is None:
            context = text

        output_text = ""
        metadata = defaultdict(list)
        n_changed_tokens = 0
        n_content_tokens = 0
        for token in analyzed_text:
            if token.pos_ in POS_MAPPING.keys():
                new_token_dict = self._process_token(token, context)
                new_token = new_token_dict["new_token"]
                output_text += new_token
                metadata["n_synsets"].append(new_token_dict["n_synsets"])
                metadata["n_lemmas"].append(new_token_dict["n_lemmas"])
                metadata["changes"].append((token.text, new_token.strip()))
                metadata["guided_unguided_responses_match"].append(new_token_dict["guided_unguided_responses_match"])
                if new_token != token.text_with_ws:
                    n_changed_tokens += 1
                n_content_tokens += 1
            else:
                output_text += token.text_with_ws
                metadata["n_synsets"].append(0)
                metadata["n_lemmas"].append(0)
                metadata["changes"].append((token.text, token.text))
                metadata["guided_unguided_responses_match"].append("no_synsets")

        metadata["n_changed_tokens"] = n_changed_tokens
        metadata["n_content_tokens"] = n_content_tokens
        return {"text": output_text, "metadata": metadata}
    
    def _process_token(
        self, 
        token: spacy.tokens.Token, 
        context: str
    ) -> Dict[str, Any]:
        """
        Process a single token and return its varied form if applicable.

        :param token: The spaCy token to process
        :param context: Context to use for word sense disambiguation.
        """
        synsets = self.wordnet.synsets(token.lemma_, pos=POS_MAPPING[token.pos_])
        
        if not synsets:
            return {"new_token": token.text_with_ws,
                    "n_synsets": 0,
                    "n_lemmas": 0,
                    "guided_unguided_responses_match": "not applicable"}

        if len(synsets) == 1:
            synset = synsets[0]
            guided_unguided_responses_match = "single_synset"  # Only one option available
        else:
            synset, guided_unguided_responses_match = self._disambiguate_word(token.lemma_, synsets, context)
        lemmas = [lemma for lemma in synset.lemmas() if lemma != token.lemma_]
        
        # If no other lemmas are available, return original token
        if not lemmas:
            return {"new_token": token.text_with_ws, 
                    "n_synsets": len(synsets), 
                    "n_lemmas": 0,
                    "guided_unguided_responses_match": guided_unguided_responses_match}
            
        selected_lemma = random.choice(lemmas)
        
        try:
            inflected_form = self._get_inflected_form(selected_lemma, token.tag_)
            # Preserve capitalization if the original token was capitalized
            if token.text[0].isupper():
                inflected_form = inflected_form[0].upper() + inflected_form[1:]
            return {"new_token": inflected_form + token.whitespace_, 
                    "n_synsets": len(synsets), 
                    "n_lemmas": len(lemmas),
                    "guided_unguided_responses_match": guided_unguided_responses_match}
        except Exception as e:
            # Fallback to original token if inflection fails
            return {"new_token": token.text_with_ws, 
                    "n_synsets": 0, 
                    "n_lemmas": 0,
                    "guided_unguided_responses_match": guided_unguided_responses_match}
    
    def _get_inflected_form(self, lemma: str, tag: str) -> str:
        """
        Get the inflected form of a lemma based on the tag.

        :param lemma: The lemma to inflect
        :param tag: The POS tag to use for inflection
        """
        lemma_parts = lemma.split()
        # Handle multi-word lemmas by inflecting only the first word
        if len(lemma_parts) > 1:
            first_word = lemma_parts[0]
            rest_of_phrase = " ".join(lemma_parts[1:])
            inflected_first = getInflection(first_word, tag)
            if not inflected_first:
                if tag == "VBP":
                    inflected_first = getInflection(first_word, "VB")
                if not inflected_first:
                    raise ValueError(f"Could not inflect lemma '{first_word}' with tag '{tag}'")
            return f"{inflected_first[0]} {rest_of_phrase}"

        # Handle single-word lemmas
        inflected_form = getInflection(lemma, tag)
        if not inflected_form:
            if tag == "VBP":
                inflected_form = getInflection(lemma, "VB")
            if not inflected_form:
                raise ValueError(f"Could not inflect lemma '{lemma}' with tag '{tag}'")
        
        return inflected_form[0]
    
    def _disambiguate_word(
        self, 
        word: str, 
        synsets: List[wn.Synset], 
        context: str
    ) -> Tuple[wn.Synset, str]:
        """
        Select the most appropriate synset for a word based on context.

        :param word: The word to disambiguate
        :param text: The original text for context
        :param synsets: List of possible synsets
        :param context: Context to use for word sense disambiguation.
        """
        instruction = self._build_wsd_instruction(word, synsets, context)
        messages = [ChatMessage.from_user(instruction)]
        
        guided_reply = self.wsd_model.run(
            messages,
            # Restrict the model to generate only a valid synset index
            generation_kwargs={
                "extra_body": {
                    "guided_choice": [f"{idx+1}" for idx in range(len(synsets))]
                }
            }
        )
        answer_text = guided_reply["replies"][0].text

        # Get model output without guidance
        unguided_reply = self.wsd_model.run(messages)
        unguided_answer_text = unguided_reply["replies"][0].text

        guided_unguided_responses_match = "matched" if unguided_answer_text[0] == answer_text[0] else "mismatched"            

        try:
            synset_idx = int(answer_text[0]) - 1
        except ValueError:
            logging.warning(f"Invalid synset index {answer_text} for {len(synsets)} synsets")
            synset_idx = 0
        
        if not 0 <= synset_idx < len(synsets):
            # Log a warning and return the first synset
            logging.warning(f"Invalid synset index {synset_idx} for {len(synsets)} synsets")
            synset_idx = 0
            
        return synsets[synset_idx], guided_unguided_responses_match
    
    def _build_wsd_instruction(
        self, 
        word: str, 
        synsets: List[wn.Synset], 
        context: str
    ) -> str:
        """
        Build the instruction for the WSD model.

        :param word: The word to disambiguate
        :param text: The original text
        :param synsets: List of possible synsets
        :param context: Context to use for word sense disambiguation.
        """
        instruction = f"Given the word \"{word}\" in the input sentence, choose the correct meaning from the following:\n"
        for idx, synset in enumerate(synsets):
            instruction += f"{idx+1}) {synset.definition()}\n"
        instruction += f"\nGenerate only the number of the selected option. Input: \"{context}"
        return instruction
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component's configuration to a dictionary.

        :return: A dictionary containing the component's configuration parameters.
        """

        wsd_model_dict = self.wsd_model.to_dict()
        return default_to_dict(
            self, 
            wsd_model=wsd_model_dict,
            spacy_model=self.spacy_model_name, 
            wordnet_version=self.wordnet_version, 
            random_seed=self.random_seed
        )
