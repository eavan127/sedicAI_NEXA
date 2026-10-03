# Speaker 2 · Poster 2 + Live Demo Part 1 · 5:00 – 10:00 (final)

*[Italic in brackets]* = what to do. **Bold** = stress it. About 670 spoken words, which is 5 minutes at a calm pace with about 20 seconds spare.

| Section | Time | Screen |
|---|---|---|
| 2.1 Overview | 5:00 – 5:15 | Poster 2 |
| 2.2 Layer 1 · Hear it | 5:15 – 5:35 | Poster 2 |
| 2.3 Layer 2 · Understand it | 5:35 – 5:55 | Poster 2 |
| 2.4 Layer 3 · Remember it | 5:55 – 6:25 | Poster 2 |
| 2.5 Layer 4 · Improve it | 6:25 – 6:45 | Poster 2 |
| 2.6 Demo · RF Replay | 6:45 – 8:30 | NEXA app |
| 2.7 Demo · Signal Analysis | 8:30 – 8:55 | NEXA app |
| 2.8 Demo · History | 8:55 – 9:40 | NEXA app |

---

## 2.1 Overview (5:00 – 5:15)
**Key points:** offline · one computer · secure in the field · hear, understand, remember, improve

*[Switch to Poster 2. Point at the top banner.]*

Thank you, [NAME]. This poster shows the whole system we built around that model.

The key line is at the top. NEXA runs **fully offline**, on **one computer**, and no data ever leaves the machine. So it works in the field, with no internet, and the data stays secure.

It has four layers: **hear it, understand it, remember it, improve it.**

---

## 2.2 Layer 1 · Hear it (5:15 – 5:35)
**Key points:** 4 inputs · same connection · real radio plugs in later

*[Point at each box]*

**Hear it.** Signals come in four ways. A **simulated receiver**, to test without hardware. **File replay**, to play an old recording as if it's arriving live. **Upload or synthesis**, to check one single capture. And next, a **real radio**.

All four use the **same connection**. So when the real radio comes, it plugs in here, and nothing else has to change.

---

## 2.3 Layer 2 · Understand it (5:35 – 5:55)
**Key points:** model in the browser · radar/FHSS = military, jamming = hostile → ALERT · unsure → REVIEW

*[Point along Layer 2, then at ALERT and REVIEW]*

**Understand it.** This is [NAME]'s model, running right inside the browser. Then comes triage.

Radar or frequency hopping is **military**. Jamming is **hostile**. Both raise an **alert** straight away.

And if the model isn't sure, it's sent for **review**, so a person checks it. That way, the operator only spends time where it's needed.

---

## 2.4 Layer 3 · Remember it (5:55 – 6:25)
**Key points:** every result saved · important signals kept forever · routine in 2 GB, oldest deleted first · audit trail · every model version

*[Point at each box in Layer 3]*

**Remember it.** Every result is saved.

For threats, unsure calls and corrections, we keep the actual signal **forever**, as evidence. Routine civilian traffic goes into a **two-gigabyte** space, and when it's full, the **oldest is deleted first**. The result is still kept. So the disk never fills up, and we don't keep other people's signals longer than we need to.

There's also an **audit trail**, which records who did what and when, so the documentation writes itself. And a copy of **every model version** is kept.

---

## 2.5 Layer 4 · Improve it (6:25 – 6:45)
**Key points:** human in the loop · operator corrects with a reason · analyst approves · more than 15% in 7 days → retrain · analyst has the final say

*[Point at OPERATOR, then ANALYST]*

**Improve it.** This is our **human-in-the-loop** feature, and it works like a real chain of command.

The **operator** corrects a mistake and writes why. A senior **analyst** checks it and approves it. If more than **fifteen percent** of reviewed captures get corrected in a week, it's time to retrain. But a new model never goes live by itself. **The analyst makes the final call.**

Let me show you.

---

## 2.6 Live demo · RF Replay (6:45 – 8:30)
**Key points:** 4 signals at once · read the screen · name each signal · jammer + QPSK both shown · correct with name + reason

*[Switch to the app. RF Replay is open, the models are loaded, "Simulated receiver" and "Contested band" are selected.]*

This is the operator's screen. I'm using the simulated receiver, with our hardest test: a **civilian link, a radar, a frequency hopper and a jammer, all at once**. Most research tests one clean signal at a time. Real life is never that clean.

*[Click Connect & stream. Point at the log lines.]*

It connects just like a real radio, and every value is labelled **"simulated"**, so we never pretend there's hardware. Every **fifty milliseconds**, a new slice of signal comes in and gets checked.

*[Let it run 2 or 3 slices, then click Stop. The last slice stays on screen.]*

I'll pause it so we can read it.

*[Point at the waterfall]*
In this picture, **time goes left to right**, and **frequency goes up and down**. Bright means a strong signal.

*[Point at the rows below]*
Underneath, there's one row per signal type, and the colour bar at the bottom is the threat level.

*[Point from left to right]*
Here's the **QPSK** link, **civilian**.
Here's the **radar**, **military**, in orange.
Here's the **frequency hopper**, also military. See how it jumps from one frequency to another.
And here comes the **jammer**: red, **hostile**, and the banner at the top says **"threat detected"**.

*[Point at where the jammer starts while QPSK is still on]*
And right here, the jammer and our QPSK link are on at the same time, and NEXA shows **both**. So the commander knows: our link isn't lost, it's being jammed. That leads to a very different decision.

*[Point at a dashed box]*
And these dashed boxes are the right answer, so you can see that NEXA's results match.

*[Click a spot on the timeline. "Correct this detection" opens.]*
If the operator disagrees, they click on it, tick what's really there, write the reason, and add their name. It goes to the analyst before anything changes. So **every change has a name and a reason on it.**

*[Click Cancel]*

---

## 2.7 Live demo · Signal Analysis (8:30 – 8:55)
**Key points:** one 160 µs window · line = signal strength, shading = attention · you can check why it decided

*[Open Signal Analysis and move the slider to your rehearsed window]*

This is one window, **one hundred and sixty microseconds**, exactly what the model saw.

The line is the signal strength. The shading is where the model was looking. They **line up**, so it's reacting to the real signal, not the noise. That means the analyst doesn't have to just trust the answer. They can **check why**.

---

## 2.8 Live demo · History (8:55 – 9:40)
**Key points:** everything saved + filters · report builder: 4 formats, each with its use · two to-do lists · hand over

*[Open History]*

**History** keeps every analysis. You can filter by verdict, signal type, input or date, so any past event can be found in seconds.

*[Point at Report builder, open Format]*
The **report builder** lets you tick what to include and pick a format. A **PDF** for official reports, with the security marking on every page. **Excel or CSV** for analysts. **JSON** for other systems. And **SigMF**, to share the actual recordings, with the approved corrections included as labels.

*[Scroll down to the two lists]*
Further down are two to-do lists: unsure results waiting for an operator, and corrections waiting for an analyst. So **nothing gets forgotten**, and **nothing changes without approval**.

So NEXA hears, understands, and remembers. [NAME] will now show you why you can trust that record.

*[Step aside for Speaker 3]*

---

## Know before you go on stage

**Where things are in each 50 ms slice of the Contested band** (from `web/generators.js`):

| Signal | From the left edge of the timeline |
|---|---|
| QPSK (civilian) | 5% – 60% (about 2.5 – 30 ms) |
| LFM radar (military) | 20% – 55% (about 10 – 27.5 ms) |
| FHSS (military) | 35% – 80% (about 17.5 – 40 ms) |
| Jamming (hostile) | 55% – 95% (about 27.5 – 47.5 ms) |
| **QPSK + jammer overlap** | **55% – 60% only (about 27.5 – 30 ms): a very small spot. Rehearse it.** |

**Colours:** civilian = teal, military = orange, hostile = red, empty = grey.

**Checklist**
- [ ] `web\start_demo.bat` is running (without it, only PDF export works) and you're signed in.
- [ ] RF Replay: the models are loaded, Source = Simulated receiver, Case = Contested band, Dwell = 50 ms.
- [ ] Rehearse: Connect → 2 or 3 slices → **Stop**, then explain the frozen screen. Don't explain while it's moving.
- [ ] Check that the QPSK + jammer overlap actually shows on the screen. If it doesn't, press Connect again and take another slice.
- [ ] Pick a Signal Analysis window where the shading clearly sits on the burst.
- [ ] Both History to-do lists have entries. They're hidden when empty.
- [ ] If the demo breaks: "Let me show you our recorded run", then play the video and keep the same order.

**Words to avoid**

| Don't say | Say |
|---|---|
| "frequency across, time down" | "time left to right, frequency up and down" |
| "in real time" | "every fifty milliseconds" |
| "95 percent sure" | "a score of 95" |
| "it's rejected" (failed model) | "it's blocked, unless the analyst overrides it with a reason" |

**Q&A backup for your part**
- **Real time?** "Not every sample. The browser checks one 50 ms slice, then the next, and the screen shows the coverage, meaning how much of the signal was actually checked, so nothing is hidden. A faster machine would close the gap."
- **How long does 2 GB last?** "About 1,600 routine slices, roughly 80 seconds of nonstop signal. That's short on purpose, because anything important is kept forever."
- **Why delete at all?** "Civilian traffic isn't what we're looking for, and keeping it forever is a privacy risk. The result is always kept, and the deletion itself is logged."
- **Why does jamming alert below 0.89?** "Missing a jammer is far worse than a false alarm. Between 0.5 and 0.89, it raises a 'possible jamming' alert."
- **What else can the report include?** "The audit trail, the model and thresholds, and the raw files with SHA-256 fingerprints, plus a raw IQ bundle export."
- **Can the analyst disagree with the review queue?** "Yes. The analyst can mark the model as right instead of correcting it, and the raw signal is kept either way."
- **Who can approve?** "Everyone signs in with their own role. Operators can only flag and correct. Only analysts can approve, retrain, or switch models."
