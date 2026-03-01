import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_merge import merge_partials


def test_merge_partials_dedupes_repetition():
    merged = merge_partials([
        'vanakkam',
        'vanakkam',
        'vanakkam nanba',
        'vanakkam nanba',
        'eppadi irukka'
    ])
    assert merged == 'vanakkam vanakkam nanba eppadi irukka'


def test_merge_handles_empty_chunks():
    merged = merge_partials(['', '   ', 'seri'])
    assert merged == 'seri'
