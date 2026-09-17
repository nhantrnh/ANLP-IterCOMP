# SOICT 2026 submission

Springer CCIS single-column version of the IterCOMP-VN work, repositioned for a
conference audience.

## What differs from `../report.pdf`

The course report leads with reproducing IterCOMP on Vietnamese. A conference
reviewer reads that as a reproduction paper and asks where the novelty is. The
contribution that is actually new and not tied to one language is the
**effective ratio**: nominal compression ratios are not comparable across
languages, measured over four writing systems (Hindi 4.97x, Thai 3.89x,
Vietnamese 2.49x, Chinese 1.87x under cl100k_base).

So this version:

- retitles to *Beyond Nominal Ratios: Language-Aware Prompt Compression for
  Multi-hop Question Answering*
- promotes the fertility/effective-ratio material out of the appendix into
  Section 2, directly after the introduction
- rewrites the abstract so the cross-language claim leads and the Vietnamese
  evaluation grounds it
- updates MuSiQue from n=500 to n=1,500, which changes a conclusion: the
  IterCOMP-Raw margin was +3.31 crossing zero, and is now +4.19 with the
  interval clear

## Build

    pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper

`llncs.cls` and `splncs04.bst` are the official Springer files from CTAN.
`microtype` is off: LNCS Type 1 fonts are not scalable and break pdfTeX font
expansion.

## Submission checklist (SOICT 2026)

| Requirement | Status |
|---|---|
| Springer CCIS template | `llncs.cls`, single column |
| A4 | yes |
| <= 12 pages excluding references | 10 pages, References from p11 |
| No page numbers | `\pagestyle{empty}` |
| Single-blind (names included) | yes |
| PDF | yes |

## Still open

Compression itself is only run on Vietnamese and English. Fertility is measured
on four languages, so a reviewer may ask why Hindi (4.97x, the worst case) is
not compressed. No multi-hop QA dataset was found for Thai or Hindi.
