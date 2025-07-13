import os
from pathlib import Path

from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
from symspellpy import SymSpell, Verbosity
from wordfreq import top_n_list, zipf_frequency


class Default:
    resources_dir = Path(os.getcwd()).parent / 'resources'
    cache_folder = resources_dir / 'cache'
    raw_folder = resources_dir / 'cache' / 'raw'
    dictionary_path = resources_dir / 'cache' / "dictionary.txt"


def match_case(original: str, new_word: str) -> str:
    if original.isupper():
        return new_word.upper()
    elif original.istitle():
        return new_word.title()
    else:
        return new_word


class TextProcessor:
    max_edit_distance = 2
    prefix_length = 7

    _ac = _mod = None

    def __init__(self, img_path,
                 modifications=None,
                 dict_path: Path = Default.dictionary_path,
                 cache_folder: Path = Default.cache_folder,
                 raw_folder: Path = Default.raw_folder):
        if modifications is None:
            modifications = {}

        self.img_path = img_path
        self.modifications = modifications
        self.dict_path = dict_path
        self.cache_folder = cache_folder
        self.raw_folder = raw_folder

    def download_dictionary(self):
        words = top_n_list("en", n=82700)
        with open(self.dict_path, "w") as dict_file:
            for w in words:
                # convert zipf freq to rough absolute count
                freq = int(10 ** zipf_frequency(w, "en"))
                dict_file.write(f"{w} {freq}\n")

    def process_image(self, img_path):
        cache_filename = f"{img_path.stem}.pb"
        try:
            with open(self.cache_folder / cache_filename, 'rb') as f:
                (response := AnnotateImageResponse())._pb.ParseFromString(f.read())
            found_from = "local cache"
        except FileNotFoundError:
            found_from = "Google Cloud"

            with open(img_path, 'rb') as image_file:
                img_content = image_file.read()

            try:
                with open(self.raw_folder / cache_filename, 'rb') as cache_file:
                    (response := AnnotateImageResponse())._pb.ParseFromString(cache_file.read())
                found_from = "raw cache"
            except FileNotFoundError:
                client = vision.ImageAnnotatorClient()
                image = vision.Image(content=img_content)
                response = client.text_detection(image=image)

                with open(self.raw_folder / cache_filename, 'wb') as cache_file:
                    cache_file.write(response._pb.SerializeToString())

            for a in response.text_annotations[1:]:
                w = a.description
                if not w.isalpha():
                    continue

                suggestions = self.sym_spell.lookup(w.lower(), Verbosity.TOP, self.max_edit_distance)
                word = match_case(w, suggestions[0].term if suggestions else w)
                a.description = self.modifications.get(word, word)

            with open(self.cache_folder / cache_filename, 'wb') as cache_file:
                cache_file.write(response._pb.SerializeToString())

        return response, found_from

    @property
    def sym_spell(self):
        if self._ac is None:
            if not self.dict_path.exists():
                self.download_dictionary()

            self._ac = SymSpell(self.max_edit_distance, self.prefix_length)
            self._ac.load_dictionary(self.dict_path, term_index=0, count_index=1)

        return self._ac
