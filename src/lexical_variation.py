import random
from haystack import component
import spacy
from lemminflect import getInflection

import wn


pos_mapping = {
    "ADJ": "a",
    "ADV": "r",
    "NOUN": "n",
    "VERB": "v",
}

@component
class LexicalVariator:
    """
    This component is used to vary the lexical content of a text.
    It uses WordNet to find synonyms for each word in the text and randomly samples one of them.
    """
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        wordnet_version: str = "oewn:2024",
        random_seed: int = None,
    ):
        """
        Initialize the LexicalVariator component.

        :param spacy_model: The spaCy model to use.
        :param wordnet_version: The WordNet version to use.
        :param random_seed: The random seed to use.
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

        random.seed(random_seed)

    @component.output_types(text=str)
    def run(self, text: str) -> str:
        """
        Vary the lexical content of a text.

        :param text: The text to vary.
        :return: The varied text.
        """
        output_text = ""
        analyzed_text = self.spacy_model(text)
        for token in analyzed_text:
            if token.pos_ in ["ADJ", "ADV", "NOUN", "VERB"]:
                synsets = self.wordnet.synsets(token.lemma_, pos=pos_mapping[token.pos_])
                if synsets:
                    # Select the first synset
                    # (Naive approach, find better solution for word-sense disambiguation)
                    synset = synsets[0]
                    lemmas = synset.lemmas()
                    # Randomly sample one of the lemmas
                    selected_lemma = random.choice(lemmas)
                    inflected_form = getInflection(selected_lemma, token.tag_)
                    if not inflected_form:
                        if token.tag_ == "VBP":
                            inflected_form = getInflection(selected_lemma, "VB")[0]
                        else:
                            selected_lemma = selected_lemma.split()
                            inflected_form = f"{getInflection(selected_lemma[0], token.tag_)[0]} {selected_lemma[1]}"
                    else:
                        inflected_form = inflected_form[0]
                    output_text += inflected_form + token.whitespace_
                else:
                    output_text += token.text_with_ws
            else:
                output_text += token.text_with_ws

        return {"text": output_text}
