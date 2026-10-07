"""Offline tests: python -m unittest discover -s tests -v

They use two shortened EUR-Lex pages and a scripted stand-in for the model,
so no network access or API key is needed.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from eu_law_nli import config, feedback, i18n, ingest, live, usage
from eu_law_nli.corpus import Corpus, chunk_provision, reference
from eu_law_nli.engine import Engine, Session, locate
from eu_law_nli.ingest import build_corpus, raw_path
from eu_law_nli.llm import LLMError, ScriptedLLM
from eu_law_nli.parser import Block, Provision, parse_html
from eu_law_nli.registry import load_document
from eu_law_nli.retriever import BM25Retriever

FIXTURES = Path(__file__).parent / "fixtures"
OJ = (FIXTURES / "oj_sample.html").read_text(encoding="utf-8")
CONSOLIDATED = (FIXTURES / "consolidated_sample.html").read_text(encoding="utf-8")
ART1_P1 = ("This Directive establishes a harmonised framework for the regulation of "
           "electronic communications networks")


class TempData(unittest.TestCase):
    """Point every data path at a temporary folder and preload the fixtures."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self._saved = {k: getattr(config, k) for k in
                       ("DATA_DIR", "RAW_DIR", "CORPUS_DIR", "I18N_DIR", "FEEDBACK_FILE",
                        "USAGE_FILE", "USAGE_LOG", "LIVE_DIR", "LIVE_CHECK", "DOCUMENTS_DIR")}
        config.DATA_DIR = self.tmp
        config.RAW_DIR = self.tmp / "raw"
        config.CORPUS_DIR = self.tmp / "corpus"
        config.I18N_DIR = self.tmp / "i18n"
        config.FEEDBACK_FILE = self.tmp / "feedback" / "out_of_scope.jsonl"
        config.USAGE_FILE = self.tmp / "usage" / "usage.jsonl"
        config.USAGE_LOG = True
        config.LIVE_DIR = self.tmp / "live"
        config.LIVE_CHECK = False  # no network in tests; LiveTests switch it on with fakes
        config.DOCUMENTS_DIR = self.tmp / "documents"
        shutil.copytree(self._saved["DOCUMENTS_DIR"], config.DOCUMENTS_DIR)
        config.RAW_DIR.mkdir(parents=True)
        raw_path("32018L1972", "EN").write_text(OJ, encoding="utf-8")
        raw_path("02018L1972-20241018", "EN").write_text(CONSOLIDATED, encoding="utf-8")
        self.doc = load_document("eecc")

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            setattr(config, key, value)
        shutil.rmtree(self.tmp, ignore_errors=True)


class ParserTests(unittest.TestCase):
    def test_official_journal_layout(self) -> None:
        provisions = {p.id: p for p in parse_html(OJ, "32018L1972")}
        self.assertEqual(sorted(provisions), ["anx_I", "art_1", "art_2", "art_39", "rct_1"])
        art1 = provisions["art_1"]
        self.assertEqual((art1.label, art1.title), ("Article 1", "Subject matter, scope and aims"))
        self.assertEqual([b.para for b in art1.blocks], ["1", "2"])
        self.assertTrue(art1.blocks[0].text.startswith("1. " + ART1_P1))
        self.assertIn("(a) implement an internal market", art1.blocks[1].text)
        self.assertEqual(len(art1.path), 3)
        self.assertTrue(art1.path[0].startswith("PART I — FRAMEWORK"))
        self.assertEqual(art1.path[2], "CHAPTER I — Subject matter, aim and definitions")
        self.assertEqual(provisions["art_2"].blocks[1].point, "1")

    def test_footnote_markers_are_removed_from_recitals(self) -> None:
        recital = next(p for p in parse_html(OJ) if p.kind == "recital")
        self.assertTrue(recital.text.startswith(
            "Directives 2002/19/EC, 2002/20/EC and 2002/22/EC of the European Parliament"))
        self.assertNotIn("(4)", recital.text)

    def test_consolidated_layout(self) -> None:
        provisions = {p.id: p for p in parse_html(CONSOLIDATED, "02018L1972-20241018")}
        self.assertEqual(sorted(provisions), ["anx_I", "art_1", "art_39"])
        art1 = provisions["art_1"]
        self.assertEqual([b.para for b in art1.blocks], ["1", "2"])
        self.assertNotIn("▼", art1.text)
        self.assertEqual(provisions["anx_I"].label, "ANNEX I")
        self.assertTrue(provisions["anx_I"].title.startswith("LIST OF CONDITIONS"))
        self.assertEqual(provisions["art_39"].blocks[0].para, "8")

    def test_both_layouts_give_the_same_wording(self) -> None:
        oj = {p.id: p for p in parse_html(OJ)}
        cons = {p.id: p for p in parse_html(CONSOLIDATED)}
        for pid in ("art_1", "art_39", "anx_I"):
            self.assertEqual(oj[pid].text, cons[pid].text, pid)

    def test_long_block_is_split_without_losing_text(self) -> None:
        sentence = "The competent authority shall publish the decision without delay. "
        p = Provision("anx_X", "annex", "X", "ANNEX X", blocks=[Block(sentence * 120)])
        chunks = chunk_provision(p)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c.text) <= 3200 for c in chunks))
        self.assertEqual(" ".join(c.text for c in chunks), p.text.strip())


class IngestTests(TempData):
    def test_only_the_latest_version_is_used_by_default(self) -> None:
        corpus = build_corpus(self.doc, self.doc.version("latest"), "EN")
        self.assertEqual(corpus.version, "2024-10-18")
        self.assertEqual(corpus.meta["counts"], {"article": 2, "recital": 0, "annex": 1})
        self.assertEqual({p.source_celex for p in corpus.provisions}, {"02018L1972-20241018"})
        self.assertEqual(list(corpus.meta["sources_sha256"]), ["02018L1972-20241018"])

    def test_recitals_can_be_added_from_the_original_when_switched_on(self) -> None:
        self.doc.recitals_from_original = True
        corpus = build_corpus(self.doc, self.doc.version("latest"), "EN")
        self.assertEqual(corpus.meta["counts"], {"article": 2, "recital": 1, "annex": 1})
        self.assertEqual(corpus.get("rct_1").source_celex, "32018L1972")
        self.assertEqual(corpus.get("art_1").source_celex, "02018L1972-20241018")
        reloaded = Corpus.load("eecc", "2024-10-18", "EN")
        self.assertEqual(reloaded.get("art_1").text, corpus.get("art_1").text)
        self.assertEqual(len(corpus.meta["sources_sha256"]), 2)

    def test_reference_pinpoints_paragraph_and_point(self) -> None:
        corpus = build_corpus(self.doc, self.doc.version("2018-12-17"), "EN")
        self.assertEqual(reference(corpus.get("art_1"), ART1_P1), "Article 1(1)")
        self.assertEqual(reference(corpus.get("art_2"), "means transmission systems"), "Article 2(1)")
        self.assertEqual(reference(corpus.get("art_1")), "Article 1")

    def test_retriever_ranks_the_right_article_first(self) -> None:
        corpus = build_corpus(self.doc, self.doc.version("2018-12-17"), "EN")
        retriever = BM25Retriever(corpus)
        top = retriever.search("harmonised standards and essential requirements", k=3)[0]
        self.assertEqual(top.chunk.provision_id, "art_39")
        top = retriever.search_many(["definition of electronic communications network"], k=3)[0]
        self.assertEqual(top.chunk.provision_id, "art_2")


class LocateTests(unittest.TestCase):
    TEXT = "(1) ‘electronic communications network’ means transmission systems, whether or not"

    def test_ignores_spacing_and_quote_style_but_returns_stored_wording(self) -> None:
        found = locate(self.TEXT, "'electronic communications network'  means transmission systems")
        self.assertEqual(found, "‘electronic communications network’ means transmission systems")

    def test_rejects_changed_wording(self) -> None:
        self.assertIsNone(locate(self.TEXT, "electronic communications network means transmission systems"))
        self.assertIsNone(locate(self.TEXT, "means"))


def analysis(**overrides) -> dict:
    base = {
        "language_code": "en", "language_name": "English", "intent": "question",
        "in_scope": True, "standalone_question": "What is the subject matter of the Directive?",
        "search_queries": ["harmonised framework regulation electronic communications"],
        "referenced_provisions": [],
    }
    return {**base, **overrides}


class EngineTests(TempData):
    def setUp(self) -> None:
        super().setUp()
        build_corpus(self.doc, self.doc.version("latest"), "EN")

    def engine(self, responder) -> tuple[Engine, ScriptedLLM]:
        llm = ScriptedLLM(responder=responder)
        return Engine(llm, self.doc), llm

    def assert_disclaimer(self, reply, wording=i18n.STRINGS["disclaimer"]) -> None:
        self.assertIn(wording, reply.text)

    def test_answer_keeps_exact_quotes_and_drops_invented_ones(self) -> None:
        def responder(name, system, user):
            if name == "classify_message":
                return analysis()
            passage_id = user.split('reference="Article 1"')[0].rsplit('id="', 1)[1].split('"')[0]
            return {"status": "answered", "answer": "Article 1(1) sets out the subject matter.",
                    "quotes": [
                        {"passage_id": passage_id, "text": ART1_P1 + ", electronic communications services"},
                        {"passage_id": passage_id, "text": "Member States shall always grant free access to all networks."},
                    ]}

        engine, _ = self.engine(responder)
        reply = engine.respond("What is this directive about?", Session())
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(len(reply.quotes), 1)
        self.assertEqual(len(reply.dropped_quotes), 1)
        quote = reply.quotes[0]
        self.assertEqual(quote.reference, "Article 1(1)")
        self.assertTrue(quote.url.endswith("?uri=CELEX:02018L1972-20241018#art_1"))
        self.assertIn(f"“{quote.text}”", reply.text)
        self.assertNotIn("free access to all networks", reply.text)
        self.assertNotIn(i18n.STRINGS["situation_invite"], reply.text)  # no set phrase
        self.assertIn("consolidated text of 18.10.2024", reply.text)
        self.assertNotIn("32018L1972", reply.text)  # no link to the 2018 text
        self.assert_disclaimer(reply)

    def test_no_verified_quote_is_stated_openly(self) -> None:
        def responder(name, system, user):
            if name == "classify_message":
                return analysis()
            return {"status": "answered", "answer": "Summary.",
                    "quotes": [{"passage_id": "P1", "text": "Words that are not in the directive at all."}]}

        engine, _ = self.engine(responder)
        reply = engine.respond("What is this directive about?", Session())
        self.assertEqual(reply.quotes, [])
        self.assertIn(i18n.STRINGS["no_verified_quote"], reply.text)
        self.assertIn(i18n.STRINGS["see_also_heading"], reply.text)
        self.assert_disclaimer(reply)

    def test_out_of_scope_subject_is_recorded_without_asking(self) -> None:
        gdpr = {"topic": "GDPR rules on cookies", "topic_en": "GDPR rules on cookies",
                "source": "other_eu_law"}
        engine, llm = self.engine(lambda name, system, user: analysis(
            in_scope=False, standalone_question="What does the GDPR say about cookies?",
            outside_topics=[gdpr]))
        reply = engine.respond("What does the GDPR say about cookies?", Session())
        self.assertEqual(reply.kind, "out_of_scope")
        self.assertIn(i18n.format_date(date.today().isoformat()), reply.text)
        self.assertIn("Your question concerns the following subject: GDPR rules on cookies.", reply.text)
        self.assertIn("I cannot review national laws", reply.text)
        self.assertNotIn("record", reply.text.lower())  # recorded silently, for the author
        self.assert_disclaimer(reply)
        stored = feedback.read_all()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["question"], "What does the GDPR say about cookies?")
        self.assertEqual(stored[0]["reason"], "out_of_scope")
        self.assertEqual(stored[0]["topics"], [gdpr])
        self.assertEqual([c["name"] for c in llm.calls], ["classify_message"])

    def test_answer_that_needs_other_law_records_it(self) -> None:
        national = {"topic": "Estonian rules on early-termination fees",
                    "topic_en": "Estonian rules on early-termination fees",
                    "source": "national_law"}
        answer = {"status": "answered",
                  "answer": "Under the European Electronic Communications Code, ...",
                  "outside_topics": [national],
                  "quotes": [{"passage_id": "P1", "text": ART1_P1}]}
        turns = iter([analysis(), answer])
        engine, llm = self.engine(lambda name, system, user: next(turns))
        first = engine.respond("Can my Estonian provider charge a fee if I leave early?", Session())
        self.assertEqual(first.kind, "answer")
        self.assertIn("which I cannot review: Estonian rules on early-termination fees",
                      first.text)
        self.assertIn("Under the European Electronic Communications Code",
                      llm.calls[1]["system"])
        stored = feedback.read_all()
        self.assertEqual(stored[0]["reason"], "partly_outside")
        self.assertEqual(stored[0]["topics"], [national])

    def test_the_models_followup_offer_is_shown_and_read_with_the_next_message(self) -> None:
        offer = "If you tell me which service you offer, I can check which of these apply."
        answer = {"status": "answered", "answer": "Under the Code, ...", "outside_topics": [],
                  "followup": offer, "quotes": [{"passage_id": "P1", "text": ART1_P1}]}
        turns = iter([analysis(), answer, analysis(intent="greeting")])
        engine, llm = self.engine(lambda name, system, user: next(turns))
        session = Session()
        reply = engine.respond("What is this directive about?", session)
        self.assertEqual(reply.followup, offer)
        self.assertIn(offer, reply.text)
        self.assertEqual(feedback.read_all(), [])  # nothing outside the act to record
        engine.respond("Yes, please", session)
        self.assertIn(offer, llm.calls[-1]["system"])  # the intake step knows what was offered

    def test_saved_translation_is_refreshed_only_where_the_english_changed(self) -> None:
        stale = {k: f"[et] {v}" for k, v in i18n.STRINGS.items()}
        stale["_source"] = dict(i18n.STRINGS, closest_heading="Old English wording.")
        stale["disclaimer"] = "[et] reviewed by a lawyer"
        config.I18N_DIR.mkdir(parents=True, exist_ok=True)
        (config.I18N_DIR / "et.json").write_text(json.dumps(stale), encoding="utf-8")
        llm = ScriptedLLM(responder=lambda name, system, user: {"closest_heading": "[et] new"})
        strings = i18n.Localiser(llm).strings("et", "Estonian")
        self.assertEqual(strings["closest_heading"], "[et] new")
        self.assertEqual(strings["disclaimer"], "[et] reviewed by a lawyer")
        self.assertIn('"closest_heading"', llm.calls[0]["user"])
        self.assertNotIn('"disclaimer"', llm.calls[0]["user"])

    def test_no_to_the_invitation_offers_further_help(self) -> None:
        engine, _ = self.engine(lambda *a: analysis(intent="consent_no"))
        reply = engine.respond("No thanks", Session())
        self.assertEqual(reply.kind, "decline")
        self.assertEqual(feedback.read_all(), [])
        self.assert_disclaimer(reply)

    def test_question_the_text_does_not_cover_follows_the_same_route(self) -> None:
        def responder(name, system, user):
            if name == "classify_message":
                return analysis()
            return {"status": "not_covered", "answer": "", "quotes": []}

        engine, _ = self.engine(responder)
        session = Session()
        reply = engine.respond("What fine applies in Estonia?", session)
        self.assertEqual(reply.kind, "out_of_scope")
        self.assertEqual(feedback.read_all()[0]["reason"], "not_covered")
        self.assert_disclaimer(reply)

    def test_situation_uses_the_comparison_instructions(self) -> None:
        def responder(name, system, user):
            if name == "classify_message":
                return analysis(intent="situation")
            return {"status": "answered", "answer": "Relevant provisions…", "quotes": []}

        engine, llm = self.engine(responder)
        reply = engine.respond("We run a small network and ...", Session())
        self.assertEqual(reply.kind, "situation")
        self.assertIn("The user has described a situation", llm.calls[-1]["system"])
        self.assert_disclaimer(reply)

    def test_yes_to_the_invitation_asks_for_the_situation(self) -> None:
        engine, _ = self.engine(lambda *a: analysis(intent="consent_yes"))
        reply = engine.respond("yes", Session())
        self.assertEqual(reply.kind, "prompt")
        self.assertIn(i18n.STRINGS["situation_prompt"], reply.text)
        self.assert_disclaimer(reply)

    def test_reply_wording_follows_the_users_language(self) -> None:
        estonian = {k: f"[et] {v}" for k, v in i18n.STRINGS.items()}
        estonian["out_of_scope"] = "[et] placeholders lost"  # must fall back to English

        def responder(name, system, user):
            if name == "classify_message":
                return analysis(language_code="et", language_name="Estonian")
            if name == "translate_interface":
                return estonian
            return {"status": "answered", "answer": "Vastus eesti keeles.", "quotes": [
                {"passage_id": "P1", "text": "This Article does not apply in respect of any of the essential requirements"}]}

        engine, llm = self.engine(responder)
        session = Session()
        reply = engine.respond("Millal artiklit 39 ei kohaldata?", session)
        self.assertEqual(reply.language, "et")
        self.assert_disclaimer(reply, "[et] " + i18n.STRINGS["disclaimer"])
        self.assertIn("written in Estonian", llm.calls[-1]["system"])
        self.assertIn("official EN text", reply.text)  # quotes shown in English, and said so
        saved = json.loads((config.I18N_DIR / "et.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["out_of_scope"], i18n.STRINGS["out_of_scope"])
        engine.respond("Ja artikkel 1?", session)
        self.assertEqual([c["name"] for c in llm.calls].count("translate_interface"), 1)

    def test_quotes_come_from_the_users_language_when_it_is_loaded(self) -> None:
        english = Corpus.load("eecc", "2024-10-18", "EN")
        translated = [Provision(p.id, p.kind, p.number, p.label.replace("Article", "Artikkel"),
                                p.title, p.path,
                                [Block("[et] " + b.text, b.para, b.point) for b in p.blocks],
                                p.source_celex) for p in english.provisions]
        Corpus("eecc", "2024-10-18", "ET", {}, translated).save()

        def responder(name, system, user):
            if name == "classify_message":
                return analysis(language_code="et", language_name="Estonian",
                                referenced_provisions=["art_39"])
            if name == "translate_interface":
                return dict(i18n.STRINGS)
            self.assertIn('language="ET"', user)
            return {"status": "answered", "answer": "Vastus.", "quotes": [
                {"passage_id": "P1", "text": "[et] 8. This Article does not apply"}]}

        engine, _ = self.engine(responder)
        reply = engine.respond("Millal artiklit 39 ei kohaldata?", Session())
        self.assertEqual(reply.quotes[0].reference, "Artikkel 39(8)")
        self.assertIn("/ET/TXT/HTML/", reply.quotes[0].url)
        self.assertNotIn("official EN text", reply.text)

    def test_answer_leads_with_a_plain_summary(self) -> None:
        answer = {"status": "answered", "summary": "Yes, it does.",
                  "answer": "Under the Code, Article 1(1) ...", "outside_topics": [],
                  "quotes": [{"passage_id": "P1", "text": ART1_P1}]}
        turns = iter([analysis(), answer])
        engine, _ = self.engine(lambda name, system, user: next(turns))
        reply = engine.respond("Does it set a framework?", Session())
        self.assertTrue(reply.body.startswith("**Yes, it does.**"))
        self.assertEqual(reply.summary, "Yes, it does.")
        self.assertIn(ART1_P1, reply.wording)  # quotes are kept apart for the collapsed view
        self.assertNotIn(ART1_P1, reply.details)

    def test_answer_streams_summary_and_explanation(self) -> None:
        answer = {"status": "answered", "summary": "Yes.", "answer": "Under the Code, ...",
                  "outside_topics": [], "quotes": []}
        turns = iter([analysis(), answer])
        engine, _ = self.engine(lambda name, system, user: next(turns))
        shown = []
        engine.respond("Question?", Session(), on_progress=shown.append)
        self.assertEqual(shown[-1], "Yes.\n\nUnder the Code, ...")

    def test_partial_json_fields_are_read_while_arriving(self) -> None:
        from eu_law_nli.engine import partial_field
        raw = r'{"status": "answered", "summary": "Line one\nwith \"quotes\" and ét'
        self.assertEqual(partial_field(raw, "summary"), 'Line one\nwith "quotes" and ét')
        self.assertEqual(partial_field(raw, "answer"), "")
        cut_mid_escape = '{"summary": "ends with ' + "\\"
        self.assertEqual(partial_field(cut_mid_escape, "summary"), "ends with ")

    def test_disclaimer_comes_with_the_first_reply_only(self) -> None:
        national = {"topic": "national fees", "topic_en": "national fees",
                    "source": "national_law"}
        answer = {"status": "answered", "summary": "S.", "answer": "Under the Code, ...",
                  "outside_topics": [national], "quotes": []}
        turns = iter([analysis(), dict(answer), analysis(), dict(answer)])
        engine, _ = self.engine(lambda name, system, user: next(turns))
        session = Session()
        first = engine.respond("Question one?", session)
        second = engine.respond("Question two?", session)
        self.assertEqual(first.disclaimer, i18n.STRINGS["disclaimer"])
        self.assert_disclaimer(first)
        self.assertEqual(second.disclaimer, "")
        self.assertNotIn(i18n.STRINGS["disclaimer"], second.text)
        self.assertIn("consolidated text of 18.10.2024", second.text)  # the source line stays
        self.assertIn(i18n.STRINGS["limits_notice"].split(":")[0], second.text)
        self.assertEqual(len(feedback.read_all()), 2)  # each answer's topics recorded

    def test_topics_log_deletes_entries_after_the_retention_period(self) -> None:
        feedback.record("eecc", "v", "en", "Old question?", topics=[])
        old = json.loads(config.FEEDBACK_FILE.read_text(encoding="utf-8"))
        old["received_at"] = "2020-01-01T00:00:00+00:00"
        config.FEEDBACK_FILE.write_text(json.dumps(old) + "\n", encoding="utf-8")
        feedback.record("eecc", "v", "en", "New question?", topics=[])
        self.assertEqual([e["question"] for e in feedback.read_all()], ["New question?"])

    def test_busy_service_gets_a_friendly_message(self) -> None:
        from eu_law_nli.llm import LLMBusy

        def responder(name, system, user):
            raise LLMBusy("529 overloaded_error")
        engine, _ = self.engine(responder)
        reply = engine.respond("Question?", Session())
        self.assertEqual(reply.kind, "error")
        self.assertIn(i18n.STRINGS["error_busy"], reply.text)
        self.assertNotIn("529", reply.text)
        self.assertEqual(usage.read_all()[-1]["error"], "529 overloaded_error")

    def test_every_turn_is_logged_with_the_question(self) -> None:
        answer = {"status": "answered", "answer": "Under the Code, ...", "outside_topics": [],
                  "quotes": [{"passage_id": "P1", "text": ART1_P1},
                             {"passage_id": "P1", "text": "Invented wording that is not there."}]}
        turns = iter([analysis(), answer, analysis(in_scope=False)])
        engine, _ = self.engine(lambda name, system, user: next(turns))
        session = Session(interface="web")
        engine.respond("What is this directive about?", session)
        engine.respond("And VAT?", session)
        first, second = usage.read_all()
        self.assertEqual(first["question"], "What is this directive about?")
        self.assertEqual(first["kind"], "answer")
        self.assertEqual(first["articles"], ["art_1"])
        self.assertEqual((first["quotes_kept"], first["quotes_dropped"]), (1, 1))
        self.assertEqual(first["interface"], "web")
        self.assertEqual(second["kind"], "out_of_scope")
        self.assertEqual(first["session"], second["session"])
        self.assertNotIn("Under the Code", json.dumps(first))  # replies are not stored

    def test_usage_entries_expire_after_the_retention_period(self) -> None:
        old = {"time": "2020-01-01T00:00:00+00:00", "question": "old"}
        recent = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "question": "recent"}
        config.USAGE_FILE.parent.mkdir(parents=True)
        config.USAGE_FILE.write_text(json.dumps(old) + "\n" + json.dumps(recent) + "\n",
                                     encoding="utf-8")
        usage.record({"question": "new"})  # logging a turn triggers the clean-up
        self.assertEqual([e["question"] for e in usage.read_all()], ["recent", "new"])

    def test_usage_log_can_be_switched_off(self) -> None:
        config.USAGE_LOG = False
        engine, _ = self.engine(lambda name, system, user: analysis(intent="greeting"))
        engine.respond("Hello", Session())
        self.assertFalse(config.USAGE_FILE.exists())

    def test_model_failure_still_returns_a_reply_with_the_disclaimer(self) -> None:
        def responder(name, system, user):
            raise LLMError("timeout")

        engine, _ = self.engine(responder)
        session = Session()
        reply = engine.respond("What is this directive about?", session)
        self.assertEqual(reply.kind, "error")
        self.assertIn(i18n.STRINGS["error_generic"], reply.text)
        self.assertFalse(session.disclaimer_shown)  # kept for the first real answer

    def test_missing_corpus_explains_how_to_build_it(self) -> None:
        shutil.rmtree(config.CORPUS_DIR)
        with self.assertRaises(FileNotFoundError) as ctx:
            Engine(ScriptedLLM(), self.doc)
        self.assertIn("python -m eu_law_nli.ingest eecc", str(ctx.exception))


class LiveTests(TempData):
    """The live check, with the EU servers replaced by local fakes."""

    def setUp(self) -> None:
        super().setUp()
        config.LIVE_CHECK = True
        live._checked_this_process.clear()
        build_corpus(self.doc, self.doc.version("latest"), "EN")
        self._real = (ingest.discover_consolidated, ingest.download_live)
        self.published = ["02018L1972-20181217", "02018L1972-20241018"]
        self.live_html = CONSOLIDATED
        ingest.discover_consolidated = lambda celex: list(self.published)
        ingest.download_live = lambda celex, lang: self.live_html

    def tearDown(self) -> None:
        live.stop_background("eecc")
        ingest.discover_consolidated, ingest.download_live = self._real
        super().tearDown()

    def wait_for(self, condition, seconds: float = 10) -> None:
        deadline = time.monotonic() + seconds
        while not condition():
            if time.monotonic() > deadline:
                self.fail("the background check did not finish in time")
            time.sleep(0.02)

    def test_background_check_does_not_block_and_runs_again_on_request(self) -> None:
        release = threading.Event()
        calls = []

        def slow_server(celex):
            calls.append(celex)
            release.wait(5)  # hold the first check until the test lets it go
            return list(self.published)
        ingest.discover_consolidated = slow_server
        started = time.monotonic()
        live.start_background("eecc")
        self.assertLess(time.monotonic() - started, 1)  # returned at once
        self.wait_for(lambda: live.is_checking("eecc"))
        started = time.monotonic()
        live.start_background("eecc")  # every page load does this, mid-check included
        self.assertLess(time.monotonic() - started, 1)
        release.set()
        self.wait_for(lambda: live.status("eecc").get("ok"))
        self.assertEqual(len(calls), 1)
        live.check_now("eecc")
        self.wait_for(lambda: len(calls) == 2 and not live.is_checking("eecc"))

    def test_reply_mentions_a_newer_text_adopted_mid_conversation(self) -> None:
        engine = Engine(ScriptedLLM(responder=lambda *a: analysis(intent="greeting")), "eecc")
        self.published.append("02018L1972-20260101")
        raw_path("02018L1972-20260101", "EN").write_text(CONSOLIDATED, encoding="utf-8")
        live.ensure_current("eecc", log=lambda m: None)
        reply = engine.respond("Hello", Session())
        self.assertIn("a newer version of 01.01.2026", reply.source)

    def test_unchanged_text_is_reported_up_to_date(self) -> None:
        state = live.ensure_current("eecc", log=lambda m: None)
        self.assertTrue(state["ok"])
        self.assertEqual(state["action"], "up to date")
        self.assertEqual(state["version"], "2024-10-18")

    def test_newer_version_is_adopted_and_used(self) -> None:
        self.published.append("02018L1972-20260101")
        raw_path("02018L1972-20260101", "EN").write_text(CONSOLIDATED, encoding="utf-8")
        state = live.ensure_current("eecc", log=lambda m: None)
        self.assertEqual(state["action"], "adopted 2026-01-01")
        doc = load_document("eecc")
        self.assertEqual(doc.version("latest").celex, "02018L1972-20260101")
        self.assertEqual(Corpus.available_languages("eecc", "2026-01-01"), ["EN"])
        engine = Engine(ScriptedLLM(responder=lambda *a: analysis(intent="greeting")), "eecc")
        reply = engine.respond("Hello", Session())
        self.assertIn("consolidated text of 01.01.2026", reply.text)
        self.assertIn("checked against the published text", reply.source)

    def test_changed_live_wording_triggers_a_rebuild(self) -> None:
        self.live_html = CONSOLIDATED.replace("harmonised framework", "harmonised legal framework")
        state = live.ensure_current("eecc", log=lambda m: None)
        self.assertEqual(state["action"], "refreshed")

    def test_offline_keeps_the_stored_copy_and_says_so(self) -> None:
        live.ensure_current("eecc", log=lambda m: None)  # one good check first

        def offline(celex):
            raise ConnectionError("no network")
        ingest.discover_consolidated = offline
        state = live.ensure_current("eecc", force=True, log=lambda m: None)
        self.assertFalse(state["ok"])
        self.assertTrue(state["last_ok_at"])
        engine = Engine(ScriptedLLM(responder=lambda *a: analysis(intent="greeting")), "eecc")
        reply = engine.respond("Hello", Session())
        self.assertIn("consolidated text of 18.10.2024", reply.text)
        self.assertIn("not checked against the published text since", reply.source)

    def test_check_runs_once_per_start_until_it_is_due_again(self) -> None:
        calls = []
        ingest.discover_consolidated = lambda celex: calls.append(celex) or list(self.published)
        live.ensure_current("eecc", log=lambda m: None)
        live.ensure_current("eecc", log=lambda m: None)
        self.assertEqual(len(calls), 1)
        live.ensure_current("eecc", force=True, log=lambda m: None)
        self.assertEqual(len(calls), 2)


GUIDE_MD = """# Guidelines on processing traffic data

## Retention of traffic data

Providers may keep traffic data for billing purposes for six months.
Traffic data must be erased or made anonymous when no longer needed.

## Location data

Location data other than traffic data may be processed only with consent.
"""


def add_web_document(doc_id: str = "guide", text: str = GUIDE_MD) -> None:
    """A 'web' source: a local Markdown file, registered and ingested."""
    path = config.DATA_DIR / f"{doc_id}.md"
    path.write_text(text, encoding="utf-8")
    entry = {
        "id": doc_id, "short_name": "Traffic data guidelines", "title": "Guidelines",
        "citation": "Guidelines 1/2026", "scope": "Traffic and location data.",
        "source": "web", "examples": ["How long may traffic data be kept?"],
        "versions": [{"id": "current", "kind": "document", "date": "2026-01-15",
                      "path": str(path), "format": "markdown"}],
    }
    (config.DOCUMENTS_DIR / f"{doc_id}.json").write_text(json.dumps(entry), encoding="utf-8")
    doc = load_document(doc_id)
    build_corpus(doc, doc.version("latest"), "EN")


class SourceTests(TempData):
    def test_markdown_file_is_split_into_sections_at_its_headings(self) -> None:
        add_web_document()
        corpus = Corpus.load("guide", "current", "EN")
        labels = [p.label for p in corpus.provisions]
        self.assertEqual(labels, ["Retention of traffic data", "Location data"])
        self.assertEqual(corpus.provisions[0].path, ["Guidelines on processing traffic data"])
        self.assertIn("six months", corpus.provisions[0].text)
        self.assertEqual(corpus.meta["counts"], {"section": 2})

    def test_html_page_is_split_into_sections(self) -> None:
        from eu_law_nli.sources import _sections_from_html
        html = ("<html><body><nav>menu</nav><main><h1>Guide</h1><p>Intro.</p>"
                "<h2 id='fees'>Fees</h2><p>No fee applies.</p><ul><li>Except A.</li></ul>"
                "</main></body></html>")
        sections = _sections_from_html(html)
        self.assertEqual([s.label for s in sections], ["Guide", "Fees"])
        self.assertEqual(sections[1].id, "fees")  # the page's own anchor, for links
        self.assertEqual([b.text for b in sections[1].blocks], ["No fee applies.", "Except A."])

    def test_web_document_is_answered_and_quoted_like_an_act(self) -> None:
        add_web_document()
        answer = {"status": "answered", "summary": "Six months.", "answer": "Section ...",
                  "outside_topics": [],
                  "quotes": [{"passage_id": "P1", "text": "Providers may keep traffic data for "
                                                          "billing purposes for six months."}]}
        turns = iter([analysis(standalone_question="How long may traffic data be kept?",
                               search_queries=["traffic data billing retention"]), answer])
        engine = Engine(ScriptedLLM(responder=lambda *a: next(turns)), "guide")
        reply = engine.respond("How long may traffic data be kept?", Session())
        self.assertEqual(len(reply.quotes), 1)
        self.assertEqual(reply.quotes[0].reference, "Retention of traffic data")
        self.assertEqual(reply.quotes[0].url, "")  # a local file has no public link
        self.assertIn("— Retention of traffic data", reply.wording)

    def test_changed_file_is_picked_up_by_the_live_check(self) -> None:
        add_web_document()
        config.LIVE_CHECK = True
        live._checked_this_process.clear()
        self.assertEqual(live.ensure_current("guide", log=lambda m: None)["action"], "up to date")
        (config.DATA_DIR / "guide.md").write_text(GUIDE_MD.replace("six", "three"),
                                                  encoding="utf-8")
        state = live.ensure_current("guide", force=True, log=lambda m: None)
        self.assertEqual(state["action"], "refreshed")
        self.assertIn("three months", Corpus.load("guide", "current", "EN").provisions[0].text)

    def test_unknown_source_type_is_reported(self) -> None:
        from eu_law_nli.sources import SourceError, for_document
        doc = load_document("eecc")
        doc.source = "ftp"
        with self.assertRaises(SourceError):
            for_document(doc)

    def test_eu_act_is_registered_with_one_command(self) -> None:
        real = (ingest.discover_consolidated, ingest.document_date)
        ingest.discover_consolidated = lambda celex: ["02016R0679-20160504"]
        ingest.document_date = lambda celex: "2016-04-27"
        try:
            ingest.add_eurlex_document("gdpr2", "32016R0679", "GDPR", "Regulation (EU) 2016/679",
                                       "Personal data.")
        finally:
            ingest.discover_consolidated, ingest.document_date = real
        doc = load_document("gdpr2")
        self.assertEqual(doc.version("latest").celex, "02016R0679-20160504")
        self.assertEqual(doc.original.date, "2016-04-27")
        with self.assertRaises(FileExistsError):
            ingest.add_eurlex_document("gdpr2", "32016R0679", "x", "y", "z")


class LibraryTests(TempData):
    def setUp(self) -> None:
        super().setUp()
        build_corpus(self.doc, self.doc.version("latest"), "EN")
        add_web_document()
        # The fixtures are a few paragraphs long, so scores are far below the
        # threshold calibrated on full acts.
        self._min_score, config.SUGGEST_MIN_SCORE = config.SUGGEST_MIN_SCORE, 0.5

    def tearDown(self) -> None:
        config.SUGGEST_MIN_SCORE = self._min_score
        super().tearDown()

    def test_library_ranks_the_act_that_matches_best(self) -> None:
        from eu_law_nli.library import Library
        library = Library()
        self.assertEqual(sorted(library.documents), ["eecc", "guide"])  # gdpr, berec not built
        ranked = library.rank(["retention of traffic data billing"])
        self.assertEqual(ranked[0].document, "guide")
        self.assertEqual(ranked[0].provisions[0], "Retention of traffic data")
        ranked = library.rank(["harmonised framework electronic communications networks"])
        self.assertEqual(ranked[0].document, "eecc")

    def test_out_of_scope_question_points_to_the_better_act(self) -> None:
        from eu_law_nli.library import Library
        question = analysis(in_scope=False, standalone_question="How long may traffic data "
                            "be retained for billing?",
                            search_queries=["traffic data retention billing"],
                            library_act="guide")
        engine = Engine(ScriptedLLM(responder=lambda *a: question), "eecc", library=Library())
        session = Session()
        reply = engine.respond("How long may traffic data be kept?", session)
        self.assertEqual(reply.kind, "out_of_scope")
        self.assertEqual(reply.suggestion["document"], "guide")
        self.assertEqual(reply.suggestion["question"], "How long may traffic data be kept?")
        self.assertIn("another act in this library: Traffic data guidelines", reply.text)
        self.assertIn("does not deal with this question, but another act", reply.text)
        self.assertEqual(feedback.read_all(), [])  # nothing to record: the library has it

    def test_conversation_carries_over_when_the_act_changes(self) -> None:
        from eu_law_nli.engine import continue_in
        from eu_law_nli.library import Library
        answer = {"status": "answered", "summary": "Up to 6 months.", "answer": "Section ...",
                  "outside_topics": [], "quotes": []}
        def responder(name, system, user):
            if name == "classify_message":
                return analysis(language_code="de", language_name="German",
                                standalone_question="How long may traffic data be kept?",
                                search_queries=["traffic data billing retention"])
            return dict(answer) if name == "write_answer" else {}  # no German wording: English

        guide = Engine(ScriptedLLM(responder=responder), "guide", library=Library())
        first = Session(language="de", language_name="German")
        guide.respond("How long may traffic data be kept?", first)

        llm = ScriptedLLM(responder=lambda *a: analysis(intent="greeting"))
        code = Engine(llm, "eecc", library=Library())
        second = continue_in(first, code.doc.short_name, "web")
        self.assertEqual(second.language, "de")  # language kept
        self.assertTrue(second.disclaimer_shown)  # the disclaimer is not repeated
        self.assertNotEqual(second.id, first.id)
        code.respond("And does that also apply to operators?", second)
        prompt = llm.calls[0]["user"]
        self.assertIn("USER: How long may traffic data be kept?", prompt)  # the earlier exchange
        self.assertIn("ASSISTANT: **Up to 6 months.**", prompt)
        self.assertIn("NOTE: The conversation now continues in another act: European "
                      "Electronic Communications Code", prompt)

    def test_shared_words_alone_do_not_trigger_a_suggestion(self) -> None:
        from eu_law_nli.library import Library
        # Keywords match the guide, but the intake step judged that no act in
        # the library covers the subject (say, a tax question about billing).
        question = analysis(in_scope=False, standalone_question="What VAT applies to billing?",
                            search_queries=["traffic data retention billing"], library_act="")
        engine = Engine(ScriptedLLM(responder=lambda *a: question), "eecc", library=Library())
        reply = engine.respond("What VAT applies to billing?", Session())
        self.assertEqual(reply.kind, "out_of_scope")
        self.assertIsNone(reply.suggestion)
        self.assertEqual(len(feedback.read_all()), 1)  # recorded for the author instead

    def test_intake_step_sees_the_other_acts_in_the_library(self) -> None:
        from eu_law_nli.library import Library
        engine, llm = None, ScriptedLLM(responder=lambda *a: analysis(intent="greeting"))
        engine = Engine(llm, "eecc", library=Library())
        engine.respond("Hello", Session())
        self.assertIn("guide: Traffic data guidelines. Traffic and location data.",
                      llm.calls[0]["system"])

    def test_answer_mentions_a_clearly_better_act_only(self) -> None:
        from eu_law_nli.library import Library
        answer = {"status": "answered", "summary": "S.", "answer": "Under the act, ...",
                  "outside_topics": [], "quotes": []}

        def ask(queries):
            turns = iter([analysis(search_queries=queries), dict(answer)])
            engine = Engine(ScriptedLLM(responder=lambda *a: next(turns)), "eecc",
                            library=Library())
            return engine.respond("Question?", Session())

        self.assertEqual(ask(["traffic data retention billing location consent"]).suggestion
                         ["document"], "guide")
        self.assertIsNone(ask(["harmonised framework electronic communications"]).suggestion)

    def test_act_names_and_examples_are_translated_with_the_fixed_wording(self) -> None:
        from eu_law_nli.library import Library
        seen = []

        def responder(name, system, user):
            if name == "translate_interface":
                keys = json.loads(user.split("\n\n", 1)[1])
                seen.extend(keys)
                return {k: f"[et] {v}" for k, v in keys.items()}
            return analysis(language_code="et", language_name="Estonian", intent="greeting")
        engine = Engine(ScriptedLLM(responder=responder), "eecc", library=Library())
        strings = engine.strings(Session(language="et", language_name="Estonian"))
        self.assertEqual(strings["guide.name"], "[et] Traffic data guidelines")
        self.assertEqual(strings["guide.example_1"], "[et] How long may traffic data be kept?")
        self.assertIn("eecc.name", seen)
        # Another set of documents keeps the translations it does not know about.
        other = i18n.Localiser(None).strings("et")
        self.assertEqual(other["disclaimer"], "[et] " + i18n.STRINGS["disclaimer"])
        saved = json.loads((config.I18N_DIR / "et.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["guide.name"], "[et] Traffic data guidelines")


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        import anthropic
        self.anthropic = anthropic
        self.saved = {k: getattr(config, k) for k in
                      ("PROVIDER", "MODEL", "BEDROCK_MODEL", "AWS_REGION", "FOUNDRY_DEPLOYMENT",
                       "FOUNDRY_AUTH")}
        self.real = (anthropic.AnthropicBedrockMantle, anthropic.AnthropicFoundry)
        self.made = []
        anthropic.AnthropicBedrockMantle = lambda **kw: self.made.append(("bedrock", kw)) or "B"
        anthropic.AnthropicFoundry = lambda **kw: self.made.append(("foundry", kw)) or "F"

    def tearDown(self) -> None:
        self.anthropic.AnthropicBedrockMantle, self.anthropic.AnthropicFoundry = self.real
        for k, v in self.saved.items():
            setattr(config, k, v)

    def test_bedrock_client_and_model_id(self) -> None:
        from eu_law_nli.llm import AnthropicLLM
        config.PROVIDER, config.AWS_REGION, config.BEDROCK_MODEL = "bedrock", "eu-central-1", ""
        llm = AnthropicLLM()
        self.assertEqual(self.made, [("bedrock", {"aws_region": "eu-central-1"})])
        self.assertEqual(llm.model, "anthropic.claude-sonnet-5-5")
        config.BEDROCK_MODEL = "eu.anthropic.claude-sonnet-5-5"
        self.assertEqual(AnthropicLLM().model, "eu.anthropic.claude-sonnet-5-5")

    def test_foundry_client_and_deployment(self) -> None:
        from eu_law_nli.llm import AnthropicLLM
        config.PROVIDER, config.FOUNDRY_DEPLOYMENT, config.FOUNDRY_AUTH = "foundry", "legal-claude", "key"
        llm = AnthropicLLM()
        self.assertEqual(self.made[0][0], "foundry")
        self.assertEqual(llm.model, "legal-claude")
        self.assertEqual(llm.provider, "foundry")

    def test_unknown_provider_is_reported(self) -> None:
        from eu_law_nli.llm import AnthropicLLM
        config.PROVIDER = "other"
        with self.assertRaises(LLMError):
            AnthropicLLM()

    def test_cost_is_estimated_for_bedrock_model_ids(self) -> None:
        self.assertAlmostEqual(usage.cost({"model": "eu.anthropic.claude-sonnet-5-5",
                                           "input_tokens": 1_000_000, "output_tokens": 0}), 2.0)


if __name__ == "__main__":
    unittest.main()
