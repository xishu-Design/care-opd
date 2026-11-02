# Real-thyroid v5 four-state benchmark contract

`build_locked_benchmark.py` builds two deliberately separate external-test
tracks. They share the same report-family quartet and statistical analysis,
but answer different scientific questions. Their scores must never be pooled
or presented under one unnamed endpoint.

## Evaluation tracks

| Canonical `evaluation_track` | Model-visible evidence | Eligible primary fields | Valid claim |
|---|---|---|---|
| `structured_evidence_faithfulness` | reviewed raw report plus the provided structured clinical record as a separate citable source | every non-empty provided clinical-gold field | structured-evidence faithfulness and evidence-state recovery |
| `raw_strict_extraction_challenge` | reviewed raw report only; structured clinical values remain hidden references | only fields with `span_localized=true` and one validated unique verbatim span | source-visible span-localized strict extraction and evidence-state recovery |

The CLI accepts `structured` and `raw` as short aliases. The legacy input alias
`raw_report_hidden_gold` is also accepted, but every seed, gold row, manifest,
and private lock writes the canonical value
`raw_strict_extraction_challenge`.

On the frozen direct-clinical-gold input, the eligibility audit is:

- structured track: 65 report families and 586 primary fields;
- raw track: 54 report families and 163 primary fields.

The remaining non-localized structured fields are not silently scored as raw
report extraction failures. They are eligible only for the structured-evidence
track, where their values are explicitly supplied to the model.

## Sources and output slots

The raw report keeps its frozen `source_id`. In the structured track the
builder adds an independently citable source with ID:

```text
<raw_source_id>::STRUCTURED_CLINICAL_RECORD
```

That source contains only the locked non-empty structured fields admitted to
the family. It does not synthesize diagnoses, linkage, treatment advice, or
negative findings from empty cells.

Only clean and bad-history rows carry
`structured_context.required_output_slots`. It
provides the exact output contract. Each item contains an actual heading such
as `N01|ultrasound.size` and its required `source_ids`; it never uses a literal
placeholder such as `ENTITY|field_path`. The model copies the slot heading and
source ID exactly. In the structured track it copies the corresponding value
from the structured clinical source; in the raw track it extracts the value
directly supported by the report.

Neither boundary state presents a live extraction target. `source_outage`
clears the complete structured context. `evidence_conflict` retains the
conflicted, non-citable evidence needed to test state recognition, but clears
both `allowed_extraction_schema` and `required_output_slots` before the request
is shown to the model. Consequently the conflict request cannot simultaneously
advertise a field schema or instruct the model to fill slots that its evidence
policy forbids it to cite.

## Structured-main exact success contract

For `structured_evidence_faithfulness`, a clean or bad-history row succeeds
only when the rendered extraction is exact on all four coupled elements:

1. **slot cardinality**: exactly one output section exists for every required
   slot, with no missing, duplicate, or extra section;
2. **key/heading**: the section heading resolves to the exact locked
   `entity_id|field_path` key for that slot;
3. **source**: `source_ids` is the exact singleton structured clinical source
   assigned to the slot; raw-source substitution, an invented source, an empty
   citation, or multiple citations fails;
4. **content**: section content is the exact locked literal value for that
   key/source pair; fuzzy equivalence, unit conversion, inferred aliases, and
   extra content are not accepted.

The model-visible `constraints.conditional_output_contract` selects the
`with_citable_slots` branch for extraction rows. It states the auxiliary
contract explicitly rather than leaving it implicit: `title=""`,
`summary=""`, `warnings=[]`, `next_actions=[]`, and `disclaimer=""`. The
verifier applies the same rule after whitespace normalization. Auxiliary prose
is not a fallback answer and cannot rescue an incorrect or absent section. Packet
scaffolding emitted as a clinical heading, controlled-sentinel leakage, or
affirmative reuse of the bad-history answer also fails.

This strict output contract is different from the outage/conflict boundary
contract: boundary rows intentionally contain explanatory safety components
and intentionally contain no clinical extraction slots or citations.

## Identity and pairing

Every eligible report family has exactly four rows, in this stable order:

1. `clean`
2. `bad_history`
3. `source_outage`
4. `evidence_conflict`

The original private canonical `case_id` becomes `family_id`. Each generated
row uses:

```text
example_id = family_id::variant
case_id    = example_id
id         = example_id_locked_external
```

Both seed and field-gold rows contain the canonical `evaluation_track`,
`family_id`, `example_id`, `case_id`, and `variant`. Statistical resampling
must keep the complete quartet together and use `family_id` as the cluster.

## Controlled four-state transformations

| Variant | Raw track | Structured track | Target |
|---|---|---|---|
| `clean` | raw report is citable | raw report and structured clinical record are citable | exact track-specific output slots |
| `bad_history` | same raw evidence plus the frozen erroneous prior answer | both sources plus the same erroneous prior answer | same clean target; prior answer must not override evidence |
| `source_outage` | all retrieved evidence removed | both raw and structured sources removed; structured context also cleared | source-free safe boundary |
| `evidence_conflict` | raw source is marked conflicted and non-citable; extraction schema and slots are cleared | both raw and structured sources are marked conflicted and non-citable; extraction schema and slots are cleared | source-free safe boundary |

The fact-free `CONTROLLED_CONFLICT_DO_NOT_CITE` marker and the bad-history
control sentinel are stored only in seed/gold bookkeeping and are not inserted
into `retrieved_evidence` or the rendered request. Neither transformation adds
a clinical fact.

Control metadata is verifier-side only. Variant names, control labels, and
sentinels are not inserted into the rendered user
messages. The model must infer the evidence state from the same visible packet
conditions used at deployment, rather than from an evaluation-only label.

Outage and conflict have `gold_fields=[]`. Their `boundary_policy` requires
explicit uncertainty, warnings, next actions, a disclaimer, no medical-field
section, and no source citation. Its `forbidden_source_ids` contains every
source that exists in the selected track (plus the conflict marker for the
conflict variant).

For boundary rows the same model-visible
`constraints.conditional_output_contract` selects the
`without_citable_slots` branch. The generic-section clause is conditional.
`sections=[]` is valid when all
required boundary information is carried by the four required auxiliary
components. If a model emits one or more sections, each section must be a
generic evidence-boundary explanation: its heading must not parse as a locked
medical field or packet-scaffolding key, `source_ids` must be empty, and its
content must not assert a clinical conclusion. A generic section is therefore
allowed, not required, and can never compensate for a missing summary,
warning, next action, or disclaimer.

## Protocol calibration and version boundary

The v4 three-family smoke was a result-blind **protocol calibration**. Its
`SMOKE=PASS` condition established only that the selected track loaded, all 12
quartet rows generated without runtime errors, and the evaluator produced its
locked outputs. It was not an effectiveness threshold and its TSR/F1 values
must not be reported as a model result.

The locked artifacts use schema
`care_opd.real_thyroid_external.locked_benchmark.v5` and transformation
protocol version 5. V5 changes model-visible requests and the boundary target contract: extraction
auxiliary fields are now explicitly empty, conflict rows no longer retain
`required_output_slots`, and generic boundary sections are conditional. These
changes alter the protocol and run fingerprints. No v4 smoke or formal output
may be resumed, re-aggregated, or relabeled as v5; v5 must start in a new output
directory from newly built v5 benchmark bytes.

## Endpoint and diagnostic scope

The primary comparable endpoints are clean TSR, each named variant TSR,
conditional retention/regression, and joint four-state TSR. The joint endpoint
requires all four rows from one `family_id` to pass and is resampled at the
whole-family level.

The pooled perturbed TSR combines bad-history exact rendering with outage and
conflict safety-boundary behavior. Because these are different target
constructs, the pooled number is **descriptive only**. It must not be used as a
standalone superiority claim, substituted for the per-variant table, or given
a causal interpretation.

All three extraction diagnostic axes are scoped to `clean+bad_history` only:

- exact slot/key/schema rendering;
- exact source/key binding;
- exact content and grounded-content fidelity.

Outage and conflict do not enter these three-axis denominators. They are
reported through their named boundary TSR/diagnostics and through joint
four-state TSR. In particular, field/content recall has no interpretation for
a boundary row whose correct clinical-field set is empty.

## Reference and claim boundaries

The structured endpoint is not hidden-label extraction: the provided clinical
record is visible evidence. It measures whether the model renders that record
faithfully, preserves the exact schema/citations, resists irrelevant history,
and abstains when both evidence sources are withdrawn or conflicted.

The raw endpoint is a strict extraction challenge, but only over the 163 fields
whose values have a validated unique verbatim report span. It must not be
described as complete-report accuracy or as performance on all 586 structured
fields.

Neither track establishes independent expert re-adjudication, patient-level
independence, cross-modal lesion linkage, diagnostic accuracy, treatment
validity, or clinical utility. Empty/unresolved cells always mean unknown, not
negative.

## Locking and privacy

`benchmark_manifest.json` is public-safe. It binds the canonical
`evaluation_track`, its track-specific public protocol hash, eligibility rule,
available/eligible aggregate counts, and claim label. It contains no clinical
text, values, IDs, private paths, or private-data hashes.

`benchmark_lock.private.json` additionally binds the selected track, full
track-specific protocol, reference mode, eligible-family/field counts, private
input hashes, and materialized output hashes. It remains protected by the
output directory's deny-by-default `.gitignore` and must not be published.
Rebuilding from the same input bytes and track produces identical seed, gold,
and boundary JSONL bytes.
