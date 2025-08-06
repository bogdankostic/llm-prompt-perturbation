from typing import Any, Dict, List, Optional, Tuple
import random

from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
import spacy

from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator
from src.custom_components.syntactic.instructions import (
    ACTIVE_TO_PASSIVE_INSTRUCTION,
    PASSIVE_TO_ACTIVE_INSTRUCTION,
    EXTRAPOSITION_INSTRUCTION,
    REVERSE_EXTRAPOSITION_INSTRUCTION,
    WH_MOVEMENT_INSTRUCTION,
    REVERSE_WH_MOVEMENT_INSTRUCTION,
    DATIVE_ALTERNATION_INSTRUCTION,
    PREP_DATIVE_ALTERNATION_INSTRUCTION,
)


SYSTEM_INSTRUCTION = """
You are an expert in linguistics, specifically in syntactic transformations.
You are given a sentence, a transformation type, and the constituents that are relevant for the transformation.
Your task is to syntactically transform the sentence using the transformation type.
Do not change the meaning of the sentence.
Do not add or change any words except for the ones that are necessary for the transformation.
The sentence should be grammatically correct and sound natural to a native speaker of English.
The sentence should contain all information that is present in the original sentence.
Keep elements that are used for layout purposes, for example 'Statement 1|', at their original position.
Output only the transformed sentence.
Output the original sentence if the transformation is not applicable.
IMPORTANT: Do not add any explanation to your output, only output the transformed sentence or the original sentence.
"""
TRANSFORMATION_INSTRUCTIONS = {
    "active_to_passive": ACTIVE_TO_PASSIVE_INSTRUCTION,
    "passive_to_active": PASSIVE_TO_ACTIVE_INSTRUCTION,
    "extraposition": EXTRAPOSITION_INSTRUCTION,
    "reverse_extraposition": REVERSE_EXTRAPOSITION_INSTRUCTION,
    "wh_movement": WH_MOVEMENT_INSTRUCTION,
    "reverse_wh_movement": REVERSE_WH_MOVEMENT_INSTRUCTION,
    "dative_alternation": DATIVE_ALTERNATION_INSTRUCTION,
    "prep_dative_alternation": PREP_DATIVE_ALTERNATION_INSTRUCTION,
}


@component
class SyntacticVariator:
    """
    A component that varies the syntactic structure of text.
    """

    def __init__(
        self,
        transformation_model: CachedOpenAIChatGenerator,
        spacy_model: str = "en_core_web_trf",
        random_seed: Optional[int] = None,
    ):
        try:
            self.spacy_model = spacy.load(spacy_model)
        except OSError:
            spacy.cli.download(spacy_model)
            self.spacy_model = spacy.load(spacy_model)
        self.transformation_model = transformation_model
        self.prompt_builder = ChatPromptBuilder()
        self.spacy_model_name = spacy_model
        self.transformation_model_name = transformation_model.model
        self.random_seed = random_seed
        random.seed(random_seed)

    @component.output_types(text=str, metadata=Dict[str, Any])
    def run(self, text: str) -> Dict[str, Any]:
        """
        Vary the syntactic structure of the sentences in the text.
        """
        doc = self.spacy_model(text)
        # Process each sentence individually
        transformed_sentences = []
        original_sentences = []
        applied_transformations = []
        for sent in doc.sents:
            applicable_transformations = self._detect_transformations(sent)
            transformation = random.choice(applicable_transformations) if applicable_transformations else ""
            if transformation:
                # Apply the transformation and add the whitespace of the last token
                transformed_sentence = self._apply_transformation(sent.text, transformation) + sent[-1].whitespace_
            else:
                transformed_sentence = sent.text_with_ws
            transformed_sentences.append(transformed_sentence)
            original_sentences.append(sent.text_with_ws)
            applied_transformations.append(transformation)

        transformed_text = "".join(transformed_sentences)
        return {"text": transformed_text, "metadata": {"transformations": applied_transformations, "original_sentences": original_sentences, "transformed_sentences": transformed_sentences}}
    
    def _detect_transformations(self, sentence: spacy.tokens.Span) -> List[Tuple[str, List[str]]]:
        """
        Detect the syntactic transformations in a sentence.
        """
        # Initialize the flags
        has_nsubj = False
        has_dobj = False
        has_nsubjpass = False
        has_auxpass = False
        has_csubj = False
        has_ccomp = False
        has_wh_word = False
        has_dative = False
        aux_position = -1
        subject_position = -1
        wh_as_subject = False
        prep_dative_alternation = False
        possible_it_extraposition = False

        # Initialize the constituents
        nsubj_constituent = ""
        dobj_constituent = ""
        nsubjpass_constituent = ""
        agent_constituent = ""
        csubj_constituent = ""
        ccomp_constituent = ""
        wh_constituent = ""
        dative_constituent = ""
        
        # Only transform sentences that contain a VERB as their head
        if sentence.root.pos_ == "VERB":
            for token_position, token in enumerate(sentence):
                # Skip tokens that are embedded in subordinate clauses
                if any(ancestor.dep_ in {"csubj", "csubjpass", "ccomp", "xcomp", "advcl", "acl", "relcl"} for ancestor in token.ancestors):
                    continue
                if token.pos_ == "AUX":
                    aux_position = token_position

                # WH-movement
                if token.tag_ in {"WDT", "WP$"}:
                    has_wh_word = True
                    # Question contains a preposition which is the head of the wh-constituent
                    if prep_token := next(token.rights, None):
                        if prep_token.dep_ == "prep":
                            wh_constituent = " ".join(cur_token.text for cur_token in prep_token.head.subtree).strip().lower()
                    if not wh_constituent:
                        wh_constituent = " ".join(cur_token.text for cur_token in token.head.subtree).strip().lower()
                    if token.dep_ in {"nsubj", "nsubjpass"}:
                        wh_as_subject = True
                elif token.tag_ in {"WP", "WRB"}:
                    has_wh_word = True
                    wh_constituent = token.text.lower()
                    if token.dep_ in {"nsubj", "nsubjpass"}:
                        wh_as_subject = True

                # Active to passive
                if token.dep_ == "nsubj":
                    has_nsubj = True
                    nsubj_constituent = " ".join(cur_token.text for cur_token in token.subtree).strip().lower()
                    possible_it_extraposition = token.text.lower() == "it"
                    subject_position = token_position
                elif token.dep_ == "dobj":
                    has_dobj = True
                    dobj_constituent = " ".join(cur_token.text for cur_token in token.subtree).strip().lower()

                # Dative alternation
                elif token.dep_ == "dative":
                    has_dative = True
                    dative_constituent = " ".join(cur_token.text for cur_token in token.subtree).strip().lower()
                    if token.pos_ == "ADP":
                        prep_dative_alternation = True

                # Passive to active
                elif token.dep_ == "nsubjpass":
                    has_nsubjpass = True
                    nsubjpass_constituent = " ".join(cur_token.text for cur_token in token.subtree).strip().lower()
                    subject_position = token_position
                elif token.dep_ == "auxpass":
                    has_auxpass = True
                elif token.dep_ == "agent":
                    if next(token.children, None):
                        agent_constituent = " ".join(cur_token.text for cur_token in token.subtree).strip().lower()

                # Extraposition
                elif token.dep_ == "csubj":
                    has_csubj = True
                    csubj_constituent = " ".join(cur_token.text for cur_token in token.subtree).strip().lower()
                elif token.dep_ == "ccomp":
                    has_ccomp = True
                    ccomp_constituent = " ".join(cur_token.text for cur_token in token.subtree).strip().lower()
                
        applicable_transformations = []
        if has_nsubj and has_dobj and not possible_it_extraposition and sentence.root.lemma_ != "have":
            relevant_constituents = [nsubj_constituent, dobj_constituent]
            applicable_transformations.append(("active_to_passive", relevant_constituents))
        if has_nsubjpass and has_auxpass and agent_constituent:
            relevant_constituents = [agent_constituent, nsubjpass_constituent]
            applicable_transformations.append(("passive_to_active", relevant_constituents))
        if has_csubj:
            relevant_constituents = [csubj_constituent]
            applicable_transformations.append(("extraposition", relevant_constituents))
        if has_ccomp and possible_it_extraposition:
            relevant_constituents = [ccomp_constituent]
            applicable_transformations.append(("reverse_extraposition", relevant_constituents))
        # Wh-movement for subjects is covert and therefore not visisble in the surface structure
        if has_wh_word and not wh_as_subject:
            # If the subject is after the auxiliary, wh-movement already took place
            if aux_position < subject_position:
                applicable_transformations.append(("reverse_wh_movement", [wh_constituent]))
            else:
                applicable_transformations.append(("wh_movement", [wh_constituent]))
        if has_dative and has_dobj:
            if prep_dative_alternation:
                applicable_transformations.append(("prep_dative_alternation", [dative_constituent]))
            else:
                applicable_transformations.append(("dative_alternation", [dative_constituent]))

        return applicable_transformations

    def _apply_transformation(self, sentence: str, transformation: str) -> str:
        """
        Apply a syntactic transformation to a sentence.
        """
        transformation_type, relevant_constituents = transformation
        transformation_instruction = ChatMessage.from_user(TRANSFORMATION_INSTRUCTIONS[transformation_type])
        messages = [ChatMessage.from_system(SYSTEM_INSTRUCTION), transformation_instruction]
        messages = self.prompt_builder.run(template=messages, sentence=sentence, constituents=relevant_constituents)["prompt"]

        model_response = self.transformation_model.run(messages=messages)["replies"][0]
        return model_response.text.strip()
