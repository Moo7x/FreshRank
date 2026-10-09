# ADR 0001: Treat later article snapshots as unavailable in strict replay

Status: Draft — agent-authored proposal, pending user review

Date: 29 September 2026

The demo implementation currently uses this working policy, but that does not make it a user-approved long-term data contract.

## Context and evidence

FreshRank's first evaluation aims to approximate what was knowable when a historical recommendation impression occurred. In the EB-NeRD demo audit, all 278,139 candidate appearances join to article records with `last_modified_time` after their impression. The supplied article table is therefore a later snapshot, not a proven historical snapshot. The dataset also contains seven-day aggregate fields and later click outcomes. Seventy training impressions include a candidate whose recorded `published_time` is after the impression; 59 have a repeated clicked ID.

## Decision

For the primary strict-replay experiment, exclude mutable article content and provider-supplied future aggregates as prediction features until historical availability can be established. A candidate's publication time may support age scoring only when it is no later than the impression. Quarantine whole impressions with a future-published candidate or duplicate clicked ID; retain valid multiple distinct clicks. Keep the official labeled validation set for evaluation only after development choices are frozen.

## Alternatives considered

- Use the supplied article snapshot directly: easier and potentially stronger, but its content may contain later edits and would not support a strict prediction-time claim. It could be a separate, clearly labeled retrospective experiment.
- Remove only the anomalous candidate or deduplicate clicks: preserves more impressions but silently changes the logged ranking task or labels.
- Drop every article with a later modification time: would remove every candidate appearance in this demo and prevent a useful first experiment.

## Consequences and revisit conditions

The first baselines have fewer usable features and 129 fewer training impressions. This is deliberate conservatism, not proof that every edited article field leaked. If historical article snapshots or reliable field-level availability times become available, revisit the exclusions and record a new ADR. The agent must still prove availability for any new feature and test that future records cannot change earlier predictions.
