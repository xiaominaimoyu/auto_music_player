import os
import tempfile
import unittest

from core.musicxml_import import read_musicxml_source
from core.score_io import import_musicxml


XML = """<?xml version='1.0'?>
<score-partwise version='4.0'>
  <part-list><score-part id='P1'><part-name>旋律</part-name></score-part></part-list>
  <part id='P1'>
    <measure number='1'>
      <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <direction><direction-type><metronome><per-minute>120</per-minute></metronome></direction-type></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>
    </measure>
  </part>
</score-partwise>
"""

TIED_XML = """<?xml version='1.0'?>
<score-partwise version='4.0'>
  <part-list><score-part id='P1'><part-name>旋律</part-name></score-part></part-list>
  <part id='P1'>
    <measure number='1'>
      <attributes><divisions>4</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><tie type='start'/></note>
    </measure>
    <measure number='2'>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><notations><tied type='stop'/></notations></note>
    </measure>
  </part>
</score-partwise>
"""


class MusicXmlImportTests(unittest.TestCase):
    def test_musicxml_to_source_and_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "score.musicxml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(XML)
            parsed = read_musicxml_source(path)
            self.assertEqual(len(parsed.song.notes), 2)
            self.assertEqual(parsed.song.bpm_hint, 120)
            result = import_musicxml(path)
            self.assertEqual(result.kind, "musicxml")
            self.assertEqual(result.source_metadata["format"], "musicxml-source")
            self.assertEqual(len(result.notes), 2)

    def test_cross_measure_tie_is_one_sustained_note(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "tied.musicxml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(TIED_XML)
            parsed = read_musicxml_source(path)
            self.assertEqual(len(parsed.song.notes), 1)
            self.assertAlmostEqual(parsed.song.notes[0].start_s, 0.0)
            self.assertAlmostEqual(parsed.song.notes[0].end_s, 1.2)
            self.assertEqual(parsed.source_metadata["notes"][0]["tie_segments"], 2)
            self.assertFalse(any("未匹配" in warning for warning in parsed.warnings))


if __name__ == "__main__":
    unittest.main()
