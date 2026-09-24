# EchoCheck Grading Rubric

## Numeric answers
When the reference answer is a single number (optionally with a unit), compare values, not strings.
Allow a relative tolerance of 0.5% for rounding. Thousands separators do not matter.
An answer that names several different values is hedging and is marked wrong even if one is right.
If both sides name a unit, the units must be compatible (m/s and metres per second are the same; km/h is not).

## Named entities and short answers
Normalise case, punctuation and articles. The answer is correct if it equals the reference or any accepted alias,
or if a short answer (up to about eight words beyond the reference) contains it.
A negated mention such as "not Paris" never counts as containing the answer.

## Accepted aliases
Aliases are added only by human adjudication (POST /adjudicate). CO2 for carbon dioxide or Bombay for Mumbai become valid only after a person confirms them.
Adjudications are stored in memory and apply to every future run of the suite.

## Open-ended answers
When a live model is configured, an LLM judge compares the candidate with the reference and the aliases, and must return a boolean with a reason.
Offline, the grader falls back to a strict fuzzy match (similarity of at least 0.9) and never uses fuzzy matching on numbers.

## Consistency
Consistency is the share of runs that agree with the majority answer for a question. It is reported separately from accuracy.
A question whose consistency stays below 100% across several runs is flaky and should be rewritten or given a tighter reference.

## Regression gate
A new run is compared with the suite's promoted baseline. Block promotion if overall accuracy drops by more than 2 percentage points,
or if any question that was always right in the baseline is now wrong in the majority of runs.
