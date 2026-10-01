# SEDIC 2026 — Phase 1 Assessment Rubric

> **Provenance.** Conversion of `SEDIC2026 - Phase 1 (Assessment Rubric).pdf`, 2 pages — page 1 is
> the Visual Track (Project Guardian), page 2 is the RF Track (Project Overwatch). **We are Track I,
> RF — section 2 below is the one that scores us.** Page 1 is included only so the two are not
> confused with each other.
>
> **Table accuracy.** Both rubrics are laid out as tables whose cells are vertically offset, and a
> plain `pdftotext` pass mis-pairs the points with their criteria — it reads as though the Benchmark
> is worth 10 and Competitive Advantage 20. The mapping below was instead recovered from per-word
> coordinates (PyMuPDF), matching each points value to the criterion on its own baseline. Both
> tracks weight identically: 30 / 25 / 10 / 20 / 15, totalling 100.

---

## 1. Project Guardian: Visual Track Evaluation Rubric

*(Not our track. Included for disambiguation only.)*

**Strategic Defence Innovation Challenge 2026**

| Evaluation Criterion | Max Points | Target Capability | Technical & Submission Requirements |
|---|---|---|---|
| Mandatory Classification | 30 | Accurately detect Civilian (e.g., container, tanker), Small Craft (e.g., yachts, fishing), and High Priority (Military) vessels. | Model must handle multi-angle classification, including frontal and aerial/satellite views. Training data must be sourced from open-source datasets. |
| Performance Benchmark | 25 | Achieve an accuracy > 80% specifically on military and threat-based classes. | Submitted as a proven performance benchmark. |
| Competitive Advantage | 10 | Distinguish between Local (Malaysian) and foreign military assets. | Models demonstrating this capability will be awarded significantly higher technical scores by the expert panel. |
| Technical Brief | 20 | Detail the dataset used, model architecture, and logic used for military classification. | Must be submitted as a PDF for Top 10 consideration. |
| Video Demonstration | 15 | Explain model functionality and results. | Must be a max 5-minute video hosted via a YouTube Channel. |

---

## 2. Project Overwatch: RF Track Evaluation Rubric

**This is our rubric.**

**Strategic Defence Innovation Challenge 2026**

| Evaluation Criterion | Max Points | Target Capability | Technical & Submission Requirements |
|---|---|---|---|
| **Mandatory Classification** | **30** | Accurately detect Civilian/Standard Comms (e.g., BPSK, QAM) and High Priority (Military/CEMA) tactical signals like Radar Pulses or FHSS bursts. | Model must identify signals from Raw IQ data in multi-condition environments, including high-SNR (clean) and low-SNR (faded/noisy) scenarios. |
| **Performance Benchmark** | **25** | Achieve an accuracy > 80% specifically on the High Priority (Military/CEMA) and Jamming classes. | Submitted as a proven **video** performance benchmark. |
| **Competitive Advantage** | **10** | Distinguish between standard communication signals and hostile CEMA interference (e.g., RF Jamming). | Models demonstrating this capability will be awarded significantly higher technical scores by the expert panel. |
| **Technical Brief** | **20** | Detail the dataset used, model architecture, and digital signal processing (DSP) logic used for tactical classification. | Must be submitted as a PDF for Top 10 consideration. |
| **Video Demonstration** | **15** | Explain model functionality and results. | Must be a max 5-minute video hosted via a YouTube Channel. |

**Total: 100 points.**

---

## Notes for the team (not part of the source document)

Everything below is our reading, kept separate from the verbatim rubric above.

1. **The video carries 40–50 of the 100 points.** Video Demonstration is only 15, but the Performance
   Benchmark's 25 is explicitly submitted *as a video* benchmark, and Competitive Advantage (10) is
   most naturally demonstrated on screen rather than asserted. This is why
   [`VIDEO_PLAN.md`](../VIDEO_PLAN.md) allocates runtime against this table instead of splitting it
   evenly between four narrators.

2. **The Technical Brief (20) is worth more than the Video Demonstration row alone (15)** — but less
   than what actually routes through the video. Both matter; neither is the junior deliverable.

3. **"Multi-condition" is a scored requirement, not a nice-to-have.** It sits in the 30-point
   Mandatory Classification row, which is why the accuracy-vs-SNR curve is non-negotiable in both the
   brief and the video.

4. **"DSP logic" is the rubric's own phrase** for what we call preprocessing (512 samples @ 3.2 MHz /
   160 µs windows, normalisation, the format contract). Use the organiser's term in the deliverables.

5. **The threshold is "accuracy," undefined.** We report recall on the three judged classes
   (LFM_RADAR, FHSS, JAMMING) and state the metric explicitly rather than letting the panel guess.

---

## Cross-references

- Verbatim rules and timeline: [`SEDIC2026_RF_track_announcement.md`](SEDIC2026_RF_track_announcement.md)
- Video plan built against this table: [`../VIDEO_PLAN.md`](../VIDEO_PLAN.md)
- Technical brief in progress: [`../TECHNICAL_BRIEF_P1-3.md`](../TECHNICAL_BRIEF_P1-3.md)
