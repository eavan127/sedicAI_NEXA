"""The data contract the chatbot depends on.

The bot only ever calls these methods. FakeStore below is for building and
testing the bot now; SqliteStore / SupabaseStore implement the same methods
later, so nothing in bot.py changes when the real database arrives.

An upload is a dict:
    {"id": int, "name": str, "time": "YYYY-MM-DD HH:MM",
     "windows": int,                      # windows analysed
     "counts": {"FHSS": 12, ...}}         # windows where each class fired
"""


class FakeStore:
    def __init__(self, uploads=None):
        self._uploads = uploads if uploads is not None else [
            {"id": 1, "name": "capture1.iq", "time": "2026-09-25 10:00", "windows": 400,
             "counts": {"FHSS": 120, "JAMMING": 30}},
            {"id": 2, "name": "harbour.iq", "time": "2026-09-25 11:30", "windows": 300,
             "counts": {"QPSK": 200, "NOISE_FLOOR": 40}},
            {"id": 3, "name": "radar_run.iq", "time": "2026-09-25 14:15", "windows": 500,
             "counts": {"LFM_RADAR": 210, "JAMMING": 15}},
        ]

    def list_uploads(self):
        return list(self._uploads)

    def get_upload(self, upload_id):
        return next((u for u in self._uploads if u["id"] == upload_id), None)

    def last_upload(self):
        return self._uploads[-1] if self._uploads else None

    def uploads_with_class(self, cls):
        return [u for u in self._uploads if u["counts"].get(cls, 0) > 0]
