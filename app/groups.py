"""Which group is the control and which is the treatment.

Alphabetical order is not an answer: for ("new_model", "old_model") it makes
the *new* variant the baseline and flips the sign of every effect, and with it
the verdict. The direction has to be decided by meaning, and when meaning is
unavailable, said out loud rather than guessed silently.
"""

from __future__ import annotations

from typing import Any

# Порядок важен: более специфичные подсказки идут раньше, чтобы "control"
# не перебивался общим "a".
_CONTROL_HINTS = (
    "control",
    "ctrl",
    "baseline",
    "base",
    "old",
    "current",
    "existing",
    "holdout",
    "контроль",
    "контрол",
    "старый",
    "старая",
    "текущ",
    "база",
)

_TREATMENT_HINTS = (
    "treatment",
    "treat",
    "variant",
    "test",
    "new",
    "exp",
    "target",
    "тест",
    "вариант",
    "новый",
    "новая",
    "целев",
)

# Односимвольные обозначения проверяются на полное совпадение: подстрока "a"
# есть почти в каждом слове.
_CONTROL_EXACT = ("a", "0", "а")
_TREATMENT_EXACT = ("b", "1", "б", "в")


def _matches(name: str, hints: tuple[str, ...], exact: tuple[str, ...]) -> bool:
    lowered = name.strip().lower()
    if lowered in exact:
        return True
    return any(hint in lowered for hint in hints)


def resolve_groups(
    names: list[str],
    control_hint: str | None = None,
    treatment_hint: str | None = None,
) -> dict[str, Any]:
    """Decide the control/treatment pair for a two-group comparison.

    Parameters
    ----------
    names : list[str]
        Group labels present in the data.
    control_hint, treatment_hint : str or None
        Explicit choice from the test config. Wins over any heuristic; an
        unknown label is ignored rather than trusted.

    Returns
    -------
    dict
        `control`, `treatment`, `reason` (how it was decided) and `ambiguous`
        (True when the fallback had to be used, so the UI can flag it).
    """
    unique = list(dict.fromkeys(str(name) for name in names))
    if len(unique) < 2:
        raise ValueError(f"need at least two groups, got {unique}")

    if control_hint and str(control_hint) in unique:
        control = str(control_hint)
        treatment = (
            str(treatment_hint)
            if treatment_hint and str(treatment_hint) in unique and str(treatment_hint) != control
            else next(name for name in unique if name != control)
        )
        return {
            "control": control,
            "treatment": treatment,
            "reason": "задано в настройках теста",
            "ambiguous": False,
        }

    if treatment_hint and str(treatment_hint) in unique:
        treatment = str(treatment_hint)
        control = next(name for name in unique if name != treatment)
        return {
            "control": control,
            "treatment": treatment,
            "reason": "задано в настройках теста",
            "ambiguous": False,
        }

    control_matches = [n for n in unique if _matches(n, _CONTROL_HINTS, _CONTROL_EXACT)]
    treatment_matches = [n for n in unique if _matches(n, _TREATMENT_HINTS, _TREATMENT_EXACT)]

    # Метка, похожая одновременно на обе роли, ничего не решает.
    control_only = [n for n in control_matches if n not in treatment_matches]
    treatment_only = [n for n in treatment_matches if n not in control_matches]

    if len(control_only) == 1 and len(unique) == 2:
        control = control_only[0]
        return {
            "control": control,
            "treatment": next(n for n in unique if n != control),
            "reason": f"«{control}» распознан как контрольная группа по названию",
            "ambiguous": False,
        }

    if len(treatment_only) == 1 and len(unique) == 2:
        treatment = treatment_only[0]
        return {
            "control": next(n for n in unique if n != treatment),
            "treatment": treatment,
            "reason": f"«{treatment}» распознан как тестовая группа по названию",
            "ambiguous": False,
        }

    ordered = sorted(unique)
    return {
        "control": ordered[0],
        "treatment": ordered[1],
        "reason": (
            f"По названиям групп определить контроль не удалось, взят алфавитный порядок: "
            f"контроль «{ordered[0]}». Если это не так, укажите контрольную группу явно — "
            "иначе знак эффекта будет перевёрнут."
        ),
        "ambiguous": True,
    }


def ordered_pair(state: Any, names: list[str]) -> tuple[str, str]:
    """Control/treatment for a node, from the assignment made at load time.

    Falls back to resolving on the spot so a node stays usable standalone.
    """
    assignment = state.get("group_assignment") or {}
    control, treatment = assignment.get("control"), assignment.get("treatment")
    if control in names and treatment in names:
        return str(control), str(treatment)

    resolved = resolve_groups(names, state.get("control_group"), state.get("treatment_group"))
    return resolved["control"], resolved["treatment"]
