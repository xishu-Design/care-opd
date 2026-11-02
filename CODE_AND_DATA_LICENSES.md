# Code and data licensing boundaries

The existing `care_opd/pyproject.toml` declares **Apache-2.0** for the training
package. This release preserves that declaration. A separate top-level license
text was not present in the provided source snapshot.

The GuardTCM application snapshot did not contain a standalone license grant.
No new blanket license is inferred for that component, its assets, or external
dependencies by publishing this repository. Rights not explicitly granted remain
with the respective rights holders.

Third-party libraries and models remain subject to their own licenses. Dependency
manifests identify the required packages; their installed source trees and model
weights are not vendored here.

Medical knowledge sources may have different terms, including noncommercial
conditions. Public availability does not by itself permit redistribution.
Dataset payloads are therefore excluded from this code release. Obtain and review
each source under its own terms before building a knowledge store or training set.
The existing component data documentation and available upstream license notices
are retained where they are part of the source documentation.
