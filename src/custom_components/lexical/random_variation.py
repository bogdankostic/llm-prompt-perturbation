import random
from typing import Optional, Dict, Any

from haystack import component, default_to_dict
import spacy
import wn
from .synonym_variation import LexicalVariator, POS_MAPPING


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
        random.seed(random_seed)
    
    def _process_token(self, token: spacy.tokens.Token) -> str:
        """
        Process a single token and return its randomly varied form if applicable.

        :param token: The spaCy token to process
        """
        # Get all synsets for the given POS
        pos = POS_MAPPING[token.pos_]
        all_synsets = list(self.wordnet.synsets(pos=pos))
        
        if not all_synsets:
            return token.text_with_ws

        # Randomly select a synset
        random_synset = random.choice(all_synsets)
        lemmas = list(random_synset.lemmas())
        
        # If no lemmas are available, return original token
        if not lemmas:
            return token.text_with_ws
            
        selected_lemma = random.choice(lemmas)
        
        try:
            inflected_form = self._get_inflected_form(selected_lemma, token.tag_)
            return inflected_form + token.whitespace_
        except Exception as e:
            # Fallback to original token if inflection fails
            return token.text_with_ws

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
