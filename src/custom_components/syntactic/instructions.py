

ACTIVE_TO_PASSIVE_INSTRUCTION = """
You are given a sentence and the agent and patient of the sentence.
Your task is to syntactically transform the sentence into passive voice by moving the patient to the subject position and placing the agent in a prepositional phrase introduced by 'by'.
Example:
Input: The chef cooked the meal.
Agent: the chef
Patient: the meal
Output: The meal was cooked by the chef.

Input: {{sentence}}
Agent: {{constituents[0]}}
Patient: {{constituents[1]}}
Output:
"""

PASSIVE_TO_ACTIVE_INSTRUCTION = """
You are given a sentence and the agent and patient of the sentence.
Your task is to syntactically transform the sentence into active voice by moving the agent to the subject position and the patient to the object position.
Example:
Input: The mouse was chased by the cat.
Agent: the cat
Patient: the mouse
Output: The cat chased the mouse.

Input: {{sentence}}
Agent: {{constituents[0]}}
Patient: {{constituents[1]}}
Output:
"""

EXTRAPOSITION_INSTRUCTION = """
You are given a sentence and the clausal subject of the sentence.
Your task is to syntactically transform the sentence using extraposition by moving the clausal subject to the end of the sentence and substituting it with the pronoun 'it' at the beginning.
Example:
Input: That she left early surprised me.
Clausal subject: that she left early
Output: It surprised me that she left early.

Input: {{sentence}}
Clausal subject: {{constituents[0]}}
Output:
"""

REVERSE_EXTRAPOSITION_INSTRUCTION = """
You are given a sentence and the extraposed clausal subject of the sentence.
Your task is to syntactically transform the sentence by undoing the extraposition, i.e. moving the clausal subject to the beginning of the sentence at the position of the pronoun 'it'.
Example:
Input: It surprised me that she left early.
Clausal subject: that she left early
Output: That she left early surprised me.

Input: {{sentence}}
Clausal subject: {{constituents[0]}}
Output:
"""

WH_MOVEMENT_INSTRUCTION = """
You are given a sentence and a wh-constituent.
Your task is to syntactically transform the sentence using wh-movement.
The wh-constituent is at its base position, move it to the beginning of the sentence.

Example:
Input: You like which of these books the most?
Wh-constituent: which of these books
Output: Which of these books do you like the most?

Input: {{sentence}}
Wh-constituent: {{constituents[0]}}
Output:
"""

REVERSE_WH_MOVEMENT_INSTRUCTION = """
You are given a sentence and a wh-constituent.
Your task is to syntactically transform the sentence by undoing the wh-movement, i.e. moving the wh-constituent from the beginning of the sentence to its base position.
Example:
Input: Which of these books do you like the most?
Wh-constituent: which of these books
Output: You like which of these books the most?

Input: {{sentence}}
Wh-constituent: {{constituents[0]}}
Output:
"""

DATIVE_ALTERNATION_INSTRUCTION = """
You are given a sentence and a dative constituent.
Your task is to syntactically transform the sentence using dative alternation, i.e. by moving the dative constituent into a prepositional phrase introduced by 'to'.
Example:
Input: I gave him the book
Dative constituent: him
Output: I gave the book to him.

Input: {{sentence}}
Dative constituent: {{constituents[0]}}
Output:
"""

PREP_DATIVE_ALTERNATION_INSTRUCTION = """
You are given a sentence and a dative constituent that appears in a prepositional phrase introduced by to.
Your task is to syntactically transform the sentence by applying dative alternation, i.e., move the dative constituent out of the prepositional phrase and into the non-prepositional object position before the theme.

Example:
Input: I gave the book to him.
Dative constituent: to him
Output: I gave him the book.

Input: {{sentence}}
Dative constituent: {{constituents[0]}}
Output:
"""