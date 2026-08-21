"""_select_best_attempt is a pure function extracted specifically so this
real bug (see docs/design.md follow-on plan) could be covered without
mocking simulate()/evaluate()/the DB: the previous version of generate()'s
retry loop always shipped whichever attempt ran *last* on exhaustion, not
whichever actually scored best against the critic's own criteria."""

from simstars.models import EndReason
from simstars.pipeline import _score_grade, _select_best_attempt


def _grade(conflict=True, escalation=True, resolution=True, dialogue=True, passes=False, reasoning="r"):
    return {
        "has_real_conflict": conflict,
        "has_escalation": escalation,
        "has_resolution": resolution,
        "dialogue_carries_the_story": dialogue,
        "passes": passes,
        "reasoning": reasoning,
    }


def _attempt(label: str, grade: dict, rounds: int = 0):
    return ([label], EndReason.TURN_BUDGET, grade, rounds)


def test_score_grade_counts_true_criteria():
    assert _score_grade(_grade(True, True, True, True)) == 4
    assert _score_grade(_grade(True, False, True, False)) == 2
    assert _score_grade(_grade(False, False, False, False)) == 0


def test_ships_the_last_attempt_when_it_passed():
    attempts = [
        _attempt("weak", _grade(False, False, False, False, passes=False)),
        _attempt("winner", _grade(True, True, True, True, passes=True)),
    ]
    events, *_ = _select_best_attempt(attempts)
    assert events == ["winner"]


def test_ships_highest_scoring_attempt_when_none_passed_not_just_the_last():
    # This is the exact bug: attempt 2 scores worst but ran last: the old
    # code shipped it unconditionally. attempt 1 scores best and must win.
    attempts = [
        _attempt("best", _grade(True, True, True, True, passes=False)),  # score 4
        _attempt("mediocre", _grade(True, True, False, False, passes=False)),  # score 2
        _attempt("worst_but_last", _grade(False, False, False, False, passes=False)),  # score 0
    ]
    events, *_ = _select_best_attempt(attempts)
    assert events == ["best"]


def test_ties_break_toward_the_later_attempt():
    attempts = [
        _attempt("earlier_tie", _grade(True, True, False, False, passes=False)),  # score 2
        _attempt("later_tie", _grade(True, True, False, False, passes=False)),  # score 2
    ]
    events, *_ = _select_best_attempt(attempts)
    assert events == ["later_tie"]


def test_single_failing_attempt_is_still_returned():
    attempts = [_attempt("only_one", _grade(False, False, False, False, passes=False))]
    events, *_ = _select_best_attempt(attempts)
    assert events == ["only_one"]
