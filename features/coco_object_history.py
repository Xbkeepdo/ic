"""Read object classes in a generated text prefix without image annotations."""

import ast
from functools import lru_cache
from pathlib import Path
import re

from nltk.stem import WordNetLemmatizer


class ObjectClassHistory:
    def __init__(self, chair_source):
        parsed = ast.parse(Path(chair_source).read_text(encoding='utf-8'))
        constants = {}
        for node in parsed.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                            'synonyms_txt', 'COCO_DOUBLE_WORDS', 'ANIMAL_WORDS', 'VEHICLE_WORDS'):
                        constants[target.id] = ast.literal_eval(node.value)
        self.lemmatizer = WordNetLemmatizer()
        self.aliases = {}
        for line in constants['synonyms_txt'].strip().splitlines():
            words = [word.strip().lower() for word in line.split(',')]
            for word in words:
                self.aliases[self.key(word)] = words[0]
        for phrase in constants['COCO_DOUBLE_WORDS']:
            self.aliases.setdefault(self.key(phrase), None)
        for word in constants['ANIMAL_WORDS']:
            for prefix in ('baby', 'adult'):
                self.aliases[self.key(prefix + ' ' + word)] = self.aliases.get(self.key(word))
        for word in constants['VEHICLE_WORDS']:
            self.aliases[self.key('passenger ' + word)] = self.aliases.get(self.key(word))
        self.aliases[self.key('toilet seat')] = self.aliases[self.key('toilet')]
        self.aliases[self.key('wine glas')] = self.aliases[self.key('wine glass')]
        self.maximum_words = max(map(len, self.aliases))

    @lru_cache(maxsize=16384)
    def lemma(self, word):
        return self.lemmatizer.lemmatize(word.lower())

    def key(self, text):
        return tuple(self.lemma(word) for word in text.split())

    def token_classes(self, tokenizer, tokens):
        result = {int(token): self.aliases[self.key(tokenizer.decode([int(token)]).strip())]
                  for token in tokens}
        assert all(result.values())
        return result

    def classes(self, text):
        # Keep hyphenated forms intact, rather than treating their components
        # as separate object mentions. This still does not infer noun/adjective
        # roles, and is only a history gate, not a replacement CHAIR evaluator.
        words = [self.lemma(word) for word in re.findall(r'[A-Za-z]+(?:-[A-Za-z]+)*', text)]
        classes, index = set(), 0
        while index < len(words):
            for size in range(min(self.maximum_words, len(words) - index), 0, -1):
                key = tuple(words[index:index + size])
                if key in self.aliases:
                    canonical = self.aliases[key]
                    if canonical is not None:
                        classes.add(canonical)
                    index += size
                    break
            else:
                index += 1
        return classes
