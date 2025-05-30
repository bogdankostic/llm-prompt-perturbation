from typing import Optional, Dict, Any
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
        context: str
    ) -> Dict[str, Any]:
        """
        Process a single token and return its varied form if applicable, using the worst fitting synset
        by recursively removing the best predicted synset.

        :param token: The spaCy token to process
        :param context: Context to use for word sense disambiguation.
        """
        # Get all synsets for the lemma
        synsets = self.wordnet.synsets(token.lemma_, pos=POS_MAPPING[token.pos_])
        
        if not synsets or len(synsets) == 1:
            return {"new_token": token.text_with_ws, "n_synsets": 0, "n_lemmas": 0}

        # Recursively remove the best predicted synset until we have only one left
        remaining_synsets = synsets.copy()
        while len(remaining_synsets) > 1:
            best_synset = self._disambiguate_word(token.lemma_, context, remaining_synsets)
            remaining_synsets.remove(best_synset)
        
        # The last remaining synset is our worst fitting one
        worst_synset = remaining_synsets[0]
        lemmas = [lemma for lemma in worst_synset.lemmas() if lemma != token.lemma_]
        
        # If no other lemmas are available, return original token
        if not lemmas:
            return {"new_token": token.text_with_ws, "n_synsets": 0, "n_lemmas": 0}
            
        selected_lemma = random.choice(lemmas)
        
        try:
            inflected_form = self._get_inflected_form(selected_lemma, token.tag_)
            # Preserve capitalization if the original token was capitalized
            if token.text[0].isupper():
                inflected_form = inflected_form[0].upper() + inflected_form[1:]
            return {"new_token": inflected_form + token.whitespace_, "n_synsets": len(synsets), "n_lemmas": len(lemmas)}
        except Exception as e:
            # Fallback to original token if inflection fails
            return {"new_token": token.text_with_ws, "n_synsets": 0, "n_lemmas": 0}
