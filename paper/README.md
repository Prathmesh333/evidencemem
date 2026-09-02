# Research paper

`main.tex` is a complete, venue-neutral research-paper draft. It uses only the verified
UIE-22K v4 confirmatory release. The title, abstract, tables, discussion, limitations,
broader-impact statement, appendix, and references are included.

The paper makes one central contribution:

> A frozen and audited experiment shows that a 440-image reliability-weighted prototype
> memory reaches 94.93% accuracy, 0.982 points below a 13,200-image kNN memory, but does
> not pass the predeclared one-point non-inferiority test or show a clear gain over an
> equal-count facility baseline.

This wording is deliberate. The point estimate is within one point, but the lower 95%
confidence bound is -1.285 points. The draft does not call the method non-inferior. It
also does not claim that reliability changes the selected prototypes in v4. The v4 run
uses the same facility-selected support set for the treatment and matched control.

## Build

Run this command inside the `paper` directory when `latexmk` is installed:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, run `pdflatex`, `bibtex`, and two additional `pdflatex` passes.

The three PDF figures in `figures/` are copied byte-for-byte from the confirmatory
release. The bibliography contains only references whose title, authors, venue or
repository, year, and persistent identifier were checked against primary sources.

## Before submission

The draft is anonymous by default. Replace the author block only after you choose a venue
and review its anonymity rules. Then apply that venue's official template without
changing the result language.

The current evidence is not sufficient for a strong generalization claim. Before an
archival submission, add a registered external-dataset experiment, a second encoder
family, byte-level storage, encoder-inclusive latency, and a direct evidence-fidelity or
human-audit measure. Do not tune the existing confirmatory split again.
