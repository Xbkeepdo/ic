"""Restrict training-token triggers to single-token COCO object word forms.

This lexical filter reads no image annotations. It does not prove that the
next generated token ends the word, or disambiguate a noun from a verb.
"""

import ast
from pathlib import Path
import re

from nltk.stem import WordNetLemmatizer


def object_token_lexicon(tokenizer, training_tokens, chair_source):
    parsed = ast.parse(Path(chair_source).read_text(encoding='utf-8'))
    synonyms = next(ast.literal_eval(node.value) for node in parsed.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'synonyms_txt'
                            for target in node.targets))
    aliases = {word.strip().lower() for line in synonyms.strip().splitlines()
               for word in line.split(',') if re.fullmatch('[A-Za-z]+', word.strip())}
    lemma = WordNetLemmatizer().lemmatize
    decoded = {}
    for token in sorted(set(map(int, training_tokens))):
        text = tokenizer.decode([token])
        word = text.strip()
        if text.startswith(' ') and re.fullmatch('[A-Za-z]+', word) and lemma(word.lower()) in aliases:
            decoded[token] = text
    return dict(token_ids=sorted(decoded), decoded=decoded,
                scope='Single-token alphabetic word forms in a standard object lexicon, intersected with old training IDs.',
                limitations='No word-boundary lookahead or contextual part-of-speech test; excludes multi-token aliases.')
