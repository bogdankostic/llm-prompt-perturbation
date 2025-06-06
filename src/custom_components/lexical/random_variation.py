import random
from typing import Optional, Dict, Any

from haystack import component, default_to_dict
import spacy
import wn
from .synonym_variation import LexicalVariator


@component
class RandomLexicalVariator(LexicalVariator):
    """
    A component that varies the lexical content of text by randomly replacing words with other words
    from random synsets in WordNet.
    
    This component uses WordNet to find random synsets for each word in the text and randomly samples
    one of their lemmas. Unlike the LexicalVariator, it doesn't use word sense disambiguation and
    instead randomly selects synsets.

    This component inherits from the LexicalVariator class and just overrides the __init__ and _process_token methods.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        wordnet_version: str = "oewn:2024",
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the RandomLexicalVariator component.

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

        self.spacy_model_name = spacy_model
        self.wordnet_version = wordnet_version
        self.random_seed = random_seed
        self.synset_per_pos = {
            "ADJ": self.wordnet.synsets(pos="a"),
            "ADV": self.wordnet.synsets(pos="r"),
            "NOUN": self.wordnet.synsets(pos="n"), 
            "VERB": self.wordnet.synsets(pos="v")
        }
        random.seed(random_seed)
    
    def _process_token(
        self, 
        token: spacy.tokens.Token,
        context: str
    ) -> Dict[str, Any]:
        """
        Process a single token and return its randomly varied form if applicable.

        :param token: The spaCy token to process
        :param context: Context to use for word sense disambiguation. Not used in random variation.
        """
        # Get all synsets for the given POS
        all_synsets = self.synset_per_pos[token.pos_]
        
        if not all_synsets:
            return {"new_token": token.text_with_ws, "n_synsets": 0, "n_lemmas": 0, "guided_unguided_responses_match": "not applicable"}

        # Randomly select a synset
        random_synset = random.choice(all_synsets)
        lemmas = random_synset.lemmas()
        
        # If no lemmas are available, return original token
        if not lemmas:
            return {"new_token": token.text_with_ws, "n_synsets": 0, "n_lemmas": 0, "guided_unguided_responses_match": "not applicable"}
            
        selected_lemma = random.choice(lemmas)
        
        try:
            inflected_form = self._get_inflected_form(selected_lemma, token.tag_)
            # Preserve capitalization if the original token was capitalized
            if token.text[0].isupper():
                inflected_form = inflected_form[0].upper() + inflected_form[1:]
            return {"new_token": inflected_form + token.whitespace_, "n_synsets": 0, "n_lemmas": 0, "guided_unguided_responses_match": "not applicable"}
        except Exception as e:
            # Fallback to original token if inflection fails
            return {"new_token": token.text_with_ws, "n_synsets": 0, "n_lemmas": 0, "guided_unguided_responses_match": "not applicable"}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component's configuration to a dictionary.

        :return: A dictionary containing the component's configuration parameters.
        """
        return default_to_dict(
            self,
            spacy_model=self.spacy_model_name,
            wordnet_version=self.wordnet_version,
            random_seed=self.random_seed
        )
