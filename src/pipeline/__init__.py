"""
Wideband pipeline: one capture in, one answer per emitter out.

    wideband IQ -> spectrogram -> detector boxes -> cut out each box's IQ
      -> classifier (60 scores per window)
      -> tracker: box -> track, scores averaged + behaviour -> final answer

  extract.py   cut one box out of the wideband IQ, centred and cleaned, and
               measure what the signal does over the box (sweeps, sidebands)
  classify.py  load the week-1 60-class model(s) and score windows
  tracker.py   join boxes across 10 ms chunks into tracks, average the scores
  resolve.py   settle the look-alike pairs (src/data/labels.py) from behaviour
  run.py       the whole chain over a capture, chunk by chunk
  capture.py   synthetic multi-chunk captures with known emitters, for tests
               and scripts/eval_pipeline.py
"""
