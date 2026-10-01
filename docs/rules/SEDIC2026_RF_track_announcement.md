# Official Announcement: SEDIC 2026 RF Track

> **Provenance.** Verbatim conversion of `SEDIC 2026 - RF track_11Aug_public.pdf` (public pack,
> 11 Aug 2026), 2 pages. Extracted with PyMuPDF. This file is the authoritative statement of the
> rules — where any document in `docs/` disagrees with it, this file wins.
>
> Only formatting has been changed: PDF bullet glyphs became Markdown bullets, and the timeline's
> whitespace padding became a table. No wording has been altered, summarised, or reordered.

---

Welcome to the SEDIC 2026 Radio Frequency (RF) Track. We are evolving the challenge to meet the
demands of modern Cyber Electromagnetic Activities (CEMA). This year, we are looking for
sophisticated AI models capable of high-stakes signal identification in complex, contested
electromagnetic environments.

---

## Phase 1: Preliminary Stage (Online Submission)

### 1. Mission Name

Project Overwatch: Cognitive Cyber Electromagnetic Activities

### 2. Format

Online Technical Proof-of-Concept (POC). Teams must develop and submit a trained AI model along with
a technical report demonstrating its capabilities.

### 3. Timeline

| # | Milestone | Date |
|---|---|---|
| 1 | Team Briefing | 14 Aug 2026 |
| 2 | Submission of Preliminary Round Deadline | **13 Sept 2026** |
| 3 | Announcement of Grand Finale Teams | 18 Sept 2026 |
| 4 | Grand Finale during CyberDSA 2026 | 07 Oct 2026 |

### 4. Technical Requirements & Mission Overview

Participants must develop a signal detection and classification model capable of multi-condition
identification, including high-SNR (clean) and low-SNR (faded/noisy) environments.

- **Training Data:** Participants are responsible for sourcing or generating their own training data
  using open-source datasets (e.g., RadioML, DeepSig) or synthetic generation tools (e.g., GNU Radio).

- **Mandatory Classifications:** Models must accurately detect and classify the following signal
  types from Raw IQ data:
  - **Civilian / Standard Comms:** Standard digital modulations (e.g., BPSK, QPSK, 16QAM, 64QAM).
  - **High Priority (MILITARY / CEMA):** Identification of tactical signals, such as Radar Pulses
    (e.g., Linear Frequency Modulation) or Frequency Hopping Spread Spectrum (FHSS) bursts.
  - **Competitive Advantage:** Models that can successfully distinguish between standard
    communication signals and hostile CEMA interference (e.g., RF Jamming) will be awarded
    significantly higher technical scores by the expert panel.

### 5. Evaluation Requirements (Submission Package)

To be considered for the Top 10 selection, teams must submit:

- **Technical Brief:** A PDF detailing the dataset used, the model architecture and the signal
  processing logic used for tactical classification.

- **A Video Performance Benchmark:** Achieve an Accuracy > 80% specifically on the High Priority
  (Military/CEMA) and Jamming classes.

- **Demonstration:** Max 5 minutes via YouTube Channel explaining the model's functionality and
  results.

---

## Phase 2: Grand Finale — CyberDSA (Top 10 Only)

Only the top 10 scoring teams from Phase 1 will be invited to the Grand Finale to showcase their
technology in a live environment to our panel of industry experts and judges.

### 1. Finalist Preparation

Selected teams must prepare the following for their allocated booth:

- **Digital Poster:** A professional visual representation of your AI pipeline, Digital Signal
  Processing (DSP) methods, and model accuracy under various noise conditions.

- **Live Demo System:** A working station capable of running your AI model in real-time.

- **Graphical User Interface (GUI):** While not mandatory, teams with a functional and intuitive GUI
  for their detection system (e.g., an interactive Spectrogram or Waterfall display highlighting
  detected CEMA threats) will receive a significant Competitive Advantage during judging.

### 2. Final Evaluation & Verification

On the final day, teams will undergo a final assessment by the judging panel:

- **Jury Presentation:** A formal pitch to a panel of experts explaining your technical approach,
  resilience to noise, and the scalability of your model for real-world CEMA operations.

---

## Notes for the team (not part of the source document)

Everything below is our reading, kept separate from the verbatim text above.

1. **The >80% benchmark is phrased as a *video* deliverable** — "A Video Performance Benchmark."
   The rubric agrees ("Submitted as a proven video performance benchmark"). The evidence must appear
   in the video, not only in the brief PDF. See [`VIDEO_PLAN.md`](../VIDEO_PLAN.md) §1.

2. **The metric is stated as "Accuracy," never recall**, and the calculation is not defined anywhere
   in this document. We report recall on the judged classes and say so explicitly.

3. **No organiser-provided IQ stream exists** — confirmed with the organiser. Nothing in this
   announcement mentions one, and there is no "classification log" requirement. Training and
   evaluation data are entirely our own. Two repo docs still say otherwise and need correcting:
   `SEDIC2026_Track1_Documentation.md:22` and `STATUS_AND_PLAN.md:64`.

4. **"Signal processing logic" is a named brief requirement** — the rubric calls it "DSP logic."
   Use the organiser's vocabulary in both the brief and the video.

5. **Phase 2 rewards a GUI** ("significant Competitive Advantage"). The OMNI console already
   satisfies this, which is worth knowing now even though it earns nothing in Phase 1.
