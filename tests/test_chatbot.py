from src.chatbot.bot import ChatBot
from src.chatbot.store import FakeStore


def bot():
    return ChatBot(FakeStore())


def test_last_upload():
    assert "radar_run.iq" in bot().ask("Show me my last upload")


def test_list_uploads_shows_all_three():
    a = bot().ask("list uploads")
    assert all(n in a for n in ("capture1.iq", "harbour.iq", "radar_run.iq"))


def test_which_uploads_had_jamming():
    a = bot().ask("Which uploads had jamming?")
    assert "capture1.iq" in a and "radar_run.iq" in a and "harbour.iq" not in a


def test_summary_of_upload_number():
    assert "harbour.iq" in bot().ask("summary of upload 2")


def test_compare_two_uploads():
    a = bot().ask("compare upload 1 and 3")
    assert "capture1.iq" in a and "radar_run.iq" in a


def test_missing_upload_is_reported_not_crashed():
    assert "could not find" in bot().ask("summary of upload 99")


def test_faq_term():
    assert "frequency-hopping" in bot().ask("what is FHSS")


def test_unknown_question_falls_back_to_help():
    assert "Sorry" in bot().ask("what is the weather")


def test_empty_store():
    b = ChatBot(FakeStore(uploads=[]))
    assert "no uploads" in b.ask("last upload")


def test_history_is_remembered_in_session():
    b = bot()
    b.ask("last upload")
    b.ask("list uploads")
    assert len(b.history) == 2
