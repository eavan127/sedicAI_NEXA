"""Fixed knowledge answers. Edit freely: one entry per term."""

_CIVILIAN = "BPSK, QPSK, 16QAM and 64QAM are ordinary civilian digital modulations."

FAQ = {
    "FHSS": "FHSS (frequency-hopping spread spectrum) is a signal that jumps between "
            "channels many times a second. It is one of the three judged classes.",
    "LFM_RADAR": "LFM_RADAR is a radar pulse whose frequency slides up during the pulse "
                 "(a chirp). It is one of the three judged classes.",
    "JAMMING": "JAMMING is deliberate interference (barrage noise, a tone or a sweep) meant "
               "to drown out other signals. It is one of the three judged classes.",
    "NOISE_FLOOR": "NOISE_FLOOR means no signal was found, only background noise. At -10 dB "
                   "a weak signal is indistinguishable from noise, so this class is limited there.",
    "BPSK": _CIVILIAN,
    "QPSK": _CIVILIAN,
    "16QAM": _CIVILIAN,
    "64QAM": _CIVILIAN,
    "SNR": "SNR (signal-to-noise ratio) is how far a signal sits above the background noise, "
           "in dB. Higher is easier to detect; at -10 dB the signal is below the noise.",
    "WINDOW": "The model looks at one window at a time: 512 samples at 3.2 MHz, which is 160 microseconds.",
    "THRESHOLD": "A class is reported when its score is above that class's threshold. "
                 "Each class has its own calibrated threshold.",
}
