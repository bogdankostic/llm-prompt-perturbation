from typing import Optional
from haystack import component
import spacy
from .synonym_variation import LexicalVariator, POS_MAPPING
import random


@component
class WrongSenseLexicalVariator(LexicalVariator):
    """
    A component that varies the lexical content of text by replacing words with synonyms from
    incorrect word senses.
    
    This component uses WordNet to find synsets for each word in the text and selects a different
    synset than the one predicted by the word sense disambiguation model. This helps test the impact
    of choosing incorrect word senses on the text's meaning.

    This component inherits from the LexicalVariator class and just overrides the _process_token method.
    """
    
    def _process_token(
        self, 
        token: spacy.tokens.Token, 
        text: str, 
        additional_context: Optional[str]
    ) -> str:
        """
        Process a single token and return its varied form if applicable, using a different synset
        than the one predicted by WSD.

        :param token: The spaCy token to process
        :param text: The original text for context
        :param additional_context: Optional additional context
        """
        # Get all synsets for the lemma
        synsets = self.wordnet.synsets(token.lemma_, pos=POS_MAPPING[token.pos_])
        
        if not synsets:
            return token.text_with_ws

        # Get the predicted synset from WSD
        predicted_synset = self._disambiguate_word(token.lemma_, text, synsets, additional_context)
        
        # Get all other synsets
        other_synsets = [s for s in synsets if s != predicted_synset]
        
        if not other_synsets:
            return token.text_with_ws

        # Randomly select a different synset
        random_synset = random.choice(other_synsets)
        lemmas = [lemma for lemma in random_synset.lemmas() if lemma != token.lemma_]
        
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
