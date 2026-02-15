"""Tests for patterns.py — regex pattern matching engine."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import patterns


class TestPatternMatch:
    """Test regex matching for common bot responses."""

    # --- ack_started ---

    def test_turkish_started_tamam(self):
        assert patterns.match("Tamam, başlıyorum") == "ack_started"

    def test_turkish_started_bakalim(self):
        assert patterns.match("Bakalım") == "ack_started"

    def test_english_started_on_it(self):
        assert patterns.match("On it") == "ack_started"

    def test_english_started_let_me(self):
        assert patterns.match("Let me work on that") == "ack_started"

    def test_english_started_ok_let_me(self):
        assert patterns.match("Ok, let me check") == "ack_started"

    def test_english_started_working_on_it(self):
        assert patterns.match("Working on it") == "ack_started"

    # --- ack_searching ---

    def test_turkish_searching(self):
        assert patterns.match("Araştırıyorum") == "ack_searching"

    def test_turkish_searching_bakiyorum(self):
        assert patterns.match("Bakıyorum şimdi") == "ack_searching"

    def test_english_searching(self):
        assert patterns.match("Searching for that info") == "ack_searching"

    def test_english_looking_up(self):
        assert patterns.match("Looking up the docs") == "ack_searching"

    def test_english_let_me_search(self):
        assert patterns.match("Let me search for that") == "ack_searching"

    # --- ack_thinking ---

    def test_turkish_thinking(self):
        assert patterns.match("Düşünüyorum") == "ack_thinking"

    def test_turkish_bir_dakika(self):
        assert patterns.match("Bir dakika") == "ack_thinking"

    def test_english_hmm(self):
        assert patterns.match("Hmm") == "ack_thinking"

    def test_english_let_me_think(self):
        assert patterns.match("Let me think") == "ack_thinking"

    def test_english_one_moment(self):
        assert patterns.match("One moment") == "ack_thinking"

    # --- ack_done ---

    def test_turkish_done(self):
        assert patterns.match("Tamamlandı") == "ack_done"

    def test_turkish_done_hallettim(self):
        assert patterns.match("Hallettim") == "ack_done"

    def test_english_done(self):
        assert patterns.match("Done") == "ack_done"

    def test_english_here_you_go(self):
        assert patterns.match("Here you go") == "ack_done"

    def test_english_all_done(self):
        assert patterns.match("All done") == "ack_done"

    # --- ack_error ---

    def test_turkish_error(self):
        assert patterns.match("Hata aldım") == "ack_error"

    def test_turkish_error_uzgunum(self):
        assert patterns.match("Üzgünüm, yapamadım") == "ack_error"

    def test_english_error(self):
        assert patterns.match("Error occurred while processing") == "ack_error"

    def test_english_sorry(self):
        assert patterns.match("Sorry, I couldn't do that") == "ack_error"

    # --- ack_progress ---

    def test_turkish_progress(self):
        assert patterns.match("2/5 tamamlandı") == "ack_progress"

    def test_english_progress_step(self):
        assert patterns.match("Step 3 of 5") == "ack_progress"

    def test_english_progress_halfway(self):
        assert patterns.match("Halfway done") == "ack_progress"

    def test_english_progress_fraction(self):
        assert patterns.match("3 of 5 done") == "ack_progress"

    # --- No match ---

    def test_no_match_unique_content(self):
        assert patterns.match("The API endpoint returns a 404 status code when the resource is not found") is None

    def test_no_match_code(self):
        assert patterns.match("def hello_world(): print('hello')") is None

    def test_no_match_empty(self):
        assert patterns.match("") is None

    def test_no_match_whitespace(self):
        assert patterns.match("   ") is None

    # --- Case insensitivity ---

    def test_case_insensitive_upper(self):
        assert patterns.match("DONE") == "ack_done"

    def test_case_insensitive_mixed(self):
        assert patterns.match("Let Me Think") == "ack_thinking"


class TestAddPatterns:
    """Test dynamic pattern addition."""

    def test_add_custom_pattern(self):
        patterns.add_patterns("ack_custom", [r"^custom response"])
        assert patterns.match("custom response here") == "ack_custom"

    def test_add_to_existing_key(self):
        original_count = len(patterns.PATTERNS.get("ack_done", []))
        patterns.add_patterns("ack_done", [r"^mission accomplished"])
        assert len(patterns.PATTERNS["ack_done"]) == original_count + 1
        assert patterns.match("Mission accomplished") == "ack_done"
