import re
import pathlib

from clldutils.text import strip_chars
from cldfbench import Dataset as BaseDataset
from cldfbench import CLDFSpec

QUOTES = '“”'


class Dataset(BaseDataset):
    dir = pathlib.Path(__file__).parent
    id = "lapollaqiang"

    def cldf_specs(self):  # A dataset must declare all CLDF sets it creates.
        return CLDFSpec(dir=self.cldf_dir, module='Generic', metadata_fname='cldf-metadata.json')

    def cmd_download(self, args):
        pass

    def cmd_makecldf(self, args):
        args.writer.cldf.add_component('LanguageTable')
        args.writer.cldf.add_component(
            'ExampleTable',
            'Text_ID',
            {'name': 'Sentence_Number', 'datatype': 'integer'},
            {'name': 'Phrase_Number', 'datatype': 'integer'},
        )
        args.writer.cldf.add_table('texts.csv', 'ID', 'Title')
        args.writer.cldf.add_foreign_key('ExampleTable', 'Text_ID', 'texts.csv', 'ID')

        args.writer.objects['LanguageTable'].append({'ID': 'qiang', 'Name':
            'Qiang', 'Glottocode': 'west2876'})

        example_number = 0
        for text_id, title, lines in iter_texts(self.raw_dir.read('Qiang-2.txt').split('\n')):
            args.writer.objects['texts.csv'].append({'ID': text_id, 'Title': title})
            text, gloss = [], []
            for igt in iter_igts(lines):
                text.extend(igt[1])
                gloss.extend(igt[2])
            for sid, sentence in enumerate(iter_sentences(zip(text, gloss)), start=1):
                for pid, phrase in enumerate(iter_phrases(sentence), start=1):
                    example_number += 1
                    args.writer.objects['ExampleTable'].append({
                        'ID': example_number,
                        'Primary_Text': ' '.join(p[0] for p in phrase),
                        'Analyzed_Word': [p[0] for p in phrase],
                        'Gloss': [p[1] for p in phrase],
                        'Text_ID': text_id,
                        'Language_ID': 'qiang',
                        'Sentence_Number': sid,
                        'Phrase_Number': pid,
                    })


def iter_phrases(chunks):
    phrase_end = ',;'
    phrase = []
    for text, gloss in chunks:
        phrase.append((text, gloss))
        if strip_chars(QUOTES, text)[-1] in phrase_end:
            yield phrase[:]
            phrase = []
    assert phrase
    yield phrase


def iter_sentences(chunks):
    sentence_end = '.!?'
    sentence = []
    for text, gloss in chunks:
        sentence.append((text, gloss))
        if strip_chars(QUOTES, text)[-1] in sentence_end:
            yield sentence[:]
            sentence = []
    assert not sentence


def iter_igts(lines):
    assert len(lines) % 3 == 0
    for text, gloss, sep in [lines[i:i+3] for i in range(0, len(lines), 3)]:
        assert not sep
        m = re.match('(?P<number>[0-9]+)\s+', text)
        assert m
        sid = m.group('number')
        text = text[m.end():].split()
        gloss = gloss.split()
        assert len(text) == len(gloss)
        yield sid, text, gloss


def iter_texts(all_lines):
    header_pattern = re.compile('Text\s+(?P<number>[0-9]+)\s*:\s+(?P<title>.+)')
    text_id, title, lines = None, None, []

    for line in all_lines:
        line = line.strip()
        header = header_pattern.match(line)
        if header:
            if text_id:
                yield text_id, title, lines
                lines = []
            text_id, title = header.group('number'), header.group('title')
            continue
        lines.append(line)

    if lines:
        yield text_id, title, lines