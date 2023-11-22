"""TODO: Add a description here."""


import csv
import json
import os

import datasets

_DESCRIPTION = """\
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
"""

_DATA_URL = "https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0"

_HOMEPAGE = "https://commonvoice.mozilla.org/en/datasets"

_LICENSE = "https://github.com/common-voice/common-voice/blob/main/LICENSE"

class CommonVoiceConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonVoice."""

    def __init__(self, name, sub_version, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.sub_version = sub_version
        self.language = kwargs.pop("language", None)
        self.date_of_snapshot = kwargs.pop("date", None)
        self.size = kwargs.pop("size", None)
        self.validated_hr_total = kwargs.pop("val_hrs", None)
        self.total_hr_total = kwargs.pop("total_hrs", None)
        self.num_of_voice = kwargs.pop("num_of_voice", None)
        description = f"Common Voice speech to text dataset in {self.language} version {self.sub_version} of {self.date_of_snapshot}. The dataset comprises {self.validated_hr_total} of validated transcribed speech data from {self.num_of_voice} speakers. The dataset has a size of {self.size}"
        super(CommonVoiceConfig, self).__init__(
            name=name, version=datasets.Version("1.0", ""), description=description, **kwargs
        )

class CommonVoice(datasets.GeneratorBasedBuilder):

    DEFAULT_WRITER_BATCH_SIZE = 1000
    # BUILDER_CONFIGS = [
    #     CommonVoiceConfig(
    #         name=lang_id,
    #         language=_LANGUAGES[lang_id]["Language"],
    #         sub_version=_LANGUAGES[lang_id]["Version"],
    #         date=_LANGUAGES[lang_id]["Date"],
    #         size=_LANGUAGES[lang_id]["Size"],
    #         val_hrs=_LANGUAGES[lang_id]["Validated_Hr_Total"],
    #         total_hrs=_LANGUAGES[lang_id]["Overall_Hr_Total"],
    #         num_of_voice=_LANGUAGES[lang_id]["Number_Of_Voice"],
    #     )
    #     for lang_id in _LANGUAGES.keys()
    # ]

    def _info(self):
        features = datasets.Features(
            {
                "client_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=48_000),
                "sentence": datasets.Value("string"),
                "up_votes": datasets.Value("int64"),
                "down_votes": datasets.Value("int64"),
                "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accent": datasets.Value("string"),
                "locale": datasets.Value("string"),
                "segment": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="sentence")],
        )

        def _split_generators(self, dl_manager):
        # Download the TAR archive that contains the audio files:
        archive_path = dl_manager.download(_DATA_URL.format(self.config.name))

        # First we locate the data using the path within the archive:
        path_to_data = "/".join(["cv-corpus-6.1-2020-12-11", self.config.name])
        path_to_clips = "/".join([path_to_data, "clips"])
        metadata_filepaths = {
            split: "/".join([path_to_data, f"{split}.tsv"])
            for split in ["train", "test", "dev", "other", "validated", "invalidated"]
        }
        # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None

        # To access the audio data from the TAR archives using the download manager,
        # we have to use the dl_manager.iter_archive method.
        #
        # This is because dl_manager.download_and_extract
        # doesn't work to stream TAR archives in streaming mode.
        # (we have to stream the files of a TAR archive one by one)
        #
        # The iter_archive method returns an iterable of (path_within_archive, file_obj) for every
        # file in the TAR archive.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["train"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["test"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["dev"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="other",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["other"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="validated",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["validated"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="invalidated",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["invalidated"],
                    "path_to_clips": path_to_clips,
                },
            ),
        ]

    def _generate_examples(self, local_extracted_archive, archive_iterator, metadata_filepath, path_to_clips):
        """Yields examples."""
        data_fields = list(self._info().features.keys())

        # audio is not a header of the csv files
        data_fields.remove("audio")
        path_idx = data_fields.index("path")

        all_field_values = {}
        metadata_found = False
        # Here we iterate over all the files within the TAR archive:
        for path, f in archive_iterator:
            # Parse the metadata CSV file
            if path == metadata_filepath:
                metadata_found = True
                lines = f.readlines()
                headline = lines[0].decode("utf-8")

                column_names = headline.strip().split("\t")
                assert (
                    column_names == data_fields
                ), f"The file should have {data_fields} as column names, but has {column_names}"
                for line in lines[1:]:
                    field_values = line.decode("utf-8").strip().split("\t")
                    # set full path for mp3 audio file
                    audio_path = "/".join([path_to_clips, field_values[path_idx]])
                    all_field_values[audio_path] = field_values
            # Else, read the audio file and yield an example
            elif path.startswith(path_to_clips):
                assert metadata_found, "Found audio clips before the metadata TSV file."
                if not all_field_values:
                    break
                if path in all_field_values:
                    # retrieve the metadata corresponding to this audio file
                    field_values = all_field_values[path]

                    # if data is incomplete, fill with empty values
                    if len(field_values) < len(data_fields):
                        field_values += (len(data_fields) - len(field_values)) * ["''"]

                    result = {key: value for key, value in zip(data_fields, field_values)}

                    # set audio feature
                    path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                    result["audio"] = {"path": path, "bytes": f.read()}
                    # set path to None if the audio file doesn't exist locally (i.e. in streaming mode)
                    result["path"] = path if local_extracted_archive else None

                    yield path, result