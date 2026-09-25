"""Question -> intent -> data function -> template reply. No model involved."""
import re

from .faq import FAQ

# words a user might type -> the class name used in the data
ALIASES = {
    "fhss": "FHSS", "hopping": "FHSS", "hop": "FHSS",
    "jamming": "JAMMING", "jammer": "JAMMING", "jam": "JAMMING",
    "radar": "LFM_RADAR", "lfm": "LFM_RADAR", "chirp": "LFM_RADAR",
    "noise": "NOISE_FLOOR",
    "bpsk": "BPSK", "qpsk": "QPSK", "16qam": "16QAM", "64qam": "64QAM",
}
EXTRA_TERMS = {"snr": "SNR", "window": "WINDOW", "threshold": "THRESHOLD"}

HELP = ("You can ask: 'show my last upload', 'list uploads', 'which uploads had jamming', "
        "'summary of upload 2', 'compare upload 1 and 3', or 'what is FHSS'.")


def _find_class(text):
    for word, cls in ALIASES.items():
        if re.search(rf"\b{re.escape(word)}", text):
            return cls
    return None


def _fmt_counts(counts):
    if not counts:
        return "nothing detected"
    return ", ".join(f"{c}: {n}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def _summary(u):
    return f"#{u['id']} {u['name']} ({u['time']}): {u['windows']} windows. Detected {_fmt_counts(u['counts'])}."


class ChatBot:
    def __init__(self, store):
        self.store = store
        self.history = []          # (question, answer) pairs kept for this session

    def ask(self, question):
        answer = self._answer(question.strip().lower())
        self.history.append((question, answer))
        return answer

    def _answer(self, q):
        if not q or q in ("help", "?"):
            return HELP

        nums = [int(n) for n in re.findall(r"\b(\d+)\b", q)]

        if "compare" in q:
            if len(nums) < 2:
                return "Tell me two upload numbers, for example 'compare upload 1 and 3'."
            a, b = self.store.get_upload(nums[0]), self.store.get_upload(nums[1])
            if not a or not b:
                return "I could not find one of those uploads. Try 'list uploads'."
            return f"A) {_summary(a)}\nB) {_summary(b)}"

        if re.search(r"\b(last|latest|recent)\b", q):
            u = self.store.last_upload()
            return _summary(u) if u else "There are no uploads yet."

        if re.search(r"\b(list|all|history)\b", q) and "upload" in q:
            ups = self.store.list_uploads()
            return "\n".join(_summary(u) for u in ups) if ups else "There are no uploads yet."

        if "summary" in q or (nums and "upload" in q):
            if not nums:
                return "Which upload number? Try 'summary of upload 2'."
            u = self.store.get_upload(nums[0])
            return _summary(u) if u else f"I could not find upload {nums[0]}."

        cls = _find_class(q)
        asks_history = re.search(r"\b(which|had|has|with|contain|contained|found|show)\b", q) and "upload" in q
        if cls and asks_history:
            ups = self.store.uploads_with_class(cls)
            if not ups:
                return f"No upload has {cls} detected."
            return f"Uploads with {cls}: " + "; ".join(
                f"#{u['id']} {u['name']} ({u['counts'][cls]} windows)" for u in ups)

        if cls and cls in FAQ:
            return FAQ[cls]
        for word, key in EXTRA_TERMS.items():
            if word in q:
                return FAQ[key]

        return "Sorry, I did not understand that. " + HELP
