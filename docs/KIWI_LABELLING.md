# Real-world HF test set from KiwiSDR: collect and label

**Purpose:** a labelled set of **real** HF signals to test NEXA against. It is never used for training
(the synthetic-only training rule). If NEXA fails on real SSB, the fix goes into the *synthetic* SSB
generator.

## 1. Collect (about 15 minutes)

One time: `git clone https://github.com/jks-prv/kiwiclient D:\tools\kiwiclient`

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/kiwi_collect.py                     # 24 targets, 30 s each, Kuala Lumpur receiver (9M2AMH)
python scripts/kiwi_collect.py --only rtty psk31   # some classes only
python scripts/kiwi_collect.py --freq 7074 --guess unknown   # one frequency you found in the browser
```

The clips go to `data/kiwi/`, named `<guess>_<kHz>_<UTC time>.wav`. They are 2-channel IQ at 12 kS/s,
about 12 kHz of band around the frequency. The script then writes `data/kiwi/labels.csv` with a
**suggested** label for each clip.

HF changes with the time of day. If a band is empty, record again in the evening, or after sunrise
for the higher bands (14 and 21 MHz).

## 2. How the suggestion is made (not by NEXA's classifier)

| Suggested | Evidence measured from the clip |
|---|---|
| `noise` | Nothing stands 6 dB above the noise |
| `cw` | Narrow (< 150 Hz) and switching on and off |
| `rtty` | Two tones of similar strength 170 / 425 / 450 / 850 Hz apart |
| `psk31` | One thin steady line **and** a PSK31 frequency (7040, 7070, 14070, 21070 kHz) |
| `ssb_voice` | 0.8 to 3.5 kHz wide, rising and falling like speech |
| `am` | Wider than 3.5 kHz with level changes (broadcast or time station) |
| `unknown` | None of the above |

The `confidence` column (0 to 1) and the `evidence` column say how sure the measurement is and why.

## 3. Confirm each label by hand

Open `data/kiwi/labels.csv` in Excel. For every row:

1. **Look and listen.** Open the receiver in the browser (http://21439.proxy2.kiwisdr.com:8073), type the
   clip's `freq_khz`, and use the mode below. Or play the clip after the live bridge has analysed it
   (Signal Pipeline page → audio).
2. **Fill in `human_label`** with one of `noise cw rtty psk31 ssb_voice am unknown`. Write a note when
   unsure. When in doubt, use **unknown**: a wrong label is worse than none.

| Label | Mode in the KiwiSDR page | You should hear / see | It is **not** this if… |
|---|---|---|---|
| `noise` | USB | Only hiss; a dark waterfall | Any steady line or voice is in the 12 kHz around it |
| `cw` | CW | Beeps of two lengths (dots and dashes) | The beeps are a constant warble (that is a data mode) |
| `rtty` | USB | A fast two-tone "diddle-diddle", two parallel lines in the waterfall | There is one line only, or many tones |
| `psk31` | USB | A soft warbling single tone; one very thin line | It is two lines (RTTY) or beeps (CW) |
| `ssb_voice` | USB above 10 MHz, LSB below | A human voice, maybe "Donald Duck" until tuned | It is music or a broadcast (that is `am`) |
| `am` | AM | Clear broadcast audio, or time-station ticks and voice | — |

3. **More than one signal in the clip?** Label the **strongest** one and add the others in `notes`.
4. Save the CSV. Rerunning `python scripts/kiwi_label.py data/kiwi` keeps your `human_label` column.

## 4. Use it

Only rows with a `human_label` count. NEXA is then run on each clip (live bridge `--replay`) and its
answer is compared with `human_label`, per class: that is the real-world score.
