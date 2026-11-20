"""Adapt the legacy vocabulary-path check without changing the frozen run."""

import argparse
import copy
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import audit_qwen3_probe_first_class as classes
from scripts import audit_qwen3_svar_first_class as original
from scripts.run_qwen3_signed_feedback import read, sha, write


def audit(root):
    path = root / 'protocol.json'
    plan = read(path)
    own = root / 'object_lexicon.json'
    source = Path(plan['source']) / 'object_lexicon.json'
    expected = plan['source_sha256'][str(own)]
    # The new run froze its copied lexicon, while the legacy checker expects
    # the source path as a second dictionary key. Both files must match that
    # already frozen digest; no checksum is derived from unverified contents.
    assert sha(own) == sha(source) == expected
    supplemented = copy.deepcopy(plan)
    supplemented['source_sha256'][str(source)] = expected
    original_read = classes.read

    def verifier_read(filename):
        return supplemented if Path(filename).resolve() == path.resolve() else original_read(filename)

    with patch.object(classes, 'read', verifier_read):
        original.audit(root)
    write(root / 'svar_audit_compatibility.json', dict(passed=True,
        verifier_sha256=sha(Path(__file__)), generation_protocol_modified=False,
        missing_legacy_source_alias=str(source) not in plan['source_sha256'],
        source_lexicon=str(source), frozen_own_lexicon=str(own), expected_sha256=expected,
        policy='Supply an in-memory source-path alias only after both files match the frozen own-copy checksum.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
