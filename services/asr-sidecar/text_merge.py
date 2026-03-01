from typing import List


def merge_partials(partials: List[str]) -> str:
    if not partials:
        return ''

    merged: List[str] = []
    for part in partials:
        normalized = (part or '').strip()
        if not normalized:
            continue
        if not merged:
            merged.append(normalized)
            continue

        if normalized == merged[-1]:
            continue

        if normalized in merged[-1]:
            continue

        merged.append(normalized)

    return ' '.join(merged).strip()
