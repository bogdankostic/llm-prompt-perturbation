import random
from typing import List, Optional, Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
import spacy
from lemminflect import getInflection
import wn

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
        wsd_model: OpenAIChatGenerator,
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

    @component.output_types(text=str)
    def run(self, text: str, additional_context: Optional[str] = None) -> Dict[str, str]:
        """
        Vary the lexical content of a text by replacing words with their synonyms.

        :param text: The text to vary.
        :param additional_context: Optional additional context to use for word sense disambiguation.
        """
        # Apply spaCy to the text for POS tagging and lemmatization
        analyzed_text = self.spacy_model(text)
        output_text = ""
        for token in analyzed_text:
            if token.pos_ in POS_MAPPING.keys():
                output_text += self._process_token(token, text, additional_context)
            else:
                output_text += token.text_with_ws

        return {"text": output_text}
    
    def _process_token(
        self, 
        token: spacy.tokens.Token, 
        text: str, 
        additional_context: Optional[str]
    ) -> str:
        """
        Process a single token and return its varied form if applicable.

        :param token: The spaCy token to process
        :param text: The original text for context
        :param additional_context: Optional additional context
        """
        synsets = self.wordnet.synsets(token.lemma_, pos=POS_MAPPING[token.pos_])
        
        if not synsets:
            return token.text_with_ws

        synset = self._disambiguate_word(token.lemma_, text, synsets, additional_context) if len(synsets) > 1 else synsets[0]
        lemmas = [lemma for lemma in synset.lemmas() if lemma != token.lemma_]
        
        # If no other lemmas are available, return original token
        if not lemmas:
            return token.text_with_ws
            
        selected_lemma = random.choice(lemmas)
        
        try:
            inflected_form = self._get_inflected_form(selected_lemma, token.tag_)
            return inflected_form + token.whitespace_
        except Exception as e:
            # Fallback to original token if inflection fails
            return token.text_with_ws
    
    def _get_inflected_form(self, lemma: str, tag: str) -> str:
        """
        Get the inflected form of a lemma based on the tag.

        :param lemma: The lemma to inflect
        :param tag: The POS tag to use for inflection
        :return: The inflected form of the lemma
        :raises ValueError: If inflection fails for the given lemma and tag
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
        text: str, 
        synsets: List[wn.Synset], 
        additional_context: Optional[str]
    ) -> wn.Synset:
        """
        Select the most appropriate synset for a word based on context.

        :param word: The word to disambiguate
        :param text: The original text for context
        :param synsets: List of possible synsets
        :param additional_context: Optional additional context
        """
        instruction = self._build_wsd_instruction(word, text, synsets, additional_context)
        messages = [ChatMessage.from_user(instruction)]
        
        reply = self.wsd_model.run(messages)
        answer_text = reply["replies"][0].text
        synset_idx = int(answer_text[0]) - 1
        
        if not 0 <= synset_idx < len(synsets):
            raise ValueError(f"Invalid synset index {synset_idx} for {len(synsets)} synsets")
            
        return synsets[synset_idx]
    
    def _build_wsd_instruction(
        self, 
        word: str, 
        text: str, 
        synsets: List[wn.Synset], 
        additional_context: Optional[str]
    ) -> str:
        """
        Build the instruction for the WSD model.

        :param word: The word to disambiguate
        :param text: The original text
        :param synsets: List of possible synsets
        :param additional_context: Optional additional context
        :return: The formatted instruction string
        """
        instruction = f"Given the word \"{word}\" in the input sentence, choose the correct meaning for the following:\n"
        for idx, synset in enumerate(synsets):
            instruction += f"{idx+1}) {synset.definition()}\n"
        instruction += f"\nGenerate only the number of the selected option. Input: \"{text}"
        instruction += f" {additional_context}\"" if additional_context else "\""
        return instruction
    