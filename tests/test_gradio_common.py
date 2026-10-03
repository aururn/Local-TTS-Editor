import unittest
from types import SimpleNamespace

from gradio_common import (
    coerce_gradio_file_path,
    parse_optional_float,
    parse_optional_int,
    parse_optional_str,
)


class OptionalInputTests(unittest.TestCase):
    def test_blank_numbers_are_unset(self):
        for parser in (parse_optional_float, parse_optional_int):
            for raw in (None, "", "  ", "none", " None ", "NONE"):
                with self.subTest(parser=parser.__name__, raw=raw):
                    self.assertIsNone(parser(raw, "scale"))

    def test_zero_and_negative_numbers_are_preserved(self):
        for raw in ("0", 0, "-2", " 42 "):
            with self.subTest(raw=raw):
                self.assertEqual(parse_optional_int(raw, "seed"), int(raw))
                self.assertEqual(parse_optional_float(raw, "scale"), float(raw))
        self.assertEqual(parse_optional_float(" 0.75 ", "scale"), 0.75)

    def test_invalid_numbers_name_the_input(self):
        for parser, kind in ((parse_optional_float, "float"), (parse_optional_int, "int")):
            for raw in ("oops", "null", "off"):
                with self.subTest(parser=parser.__name__, raw=raw):
                    with self.assertRaisesRegex(ValueError, f"seed must be an? {kind} or blank"):
                        parser(raw, "seed")
        with self.assertRaisesRegex(ValueError, "seed must be an int or blank"):
            parse_optional_int("1.5", "seed")

    def test_disabled_adapter_values_are_unset(self):
        for raw in (None, "", "  ", "none", "NULL", "off", "disable", "disabled", " base "):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_optional_str(raw))
        self.assertEqual(parse_optional_str(" ./adapter.safetensors "), "./adapter.safetensors")


class UploadedFileTests(unittest.TestCase):
    def test_empty_uploads_are_unset(self):
        for value in (None, "", "  ", {}, {"path": None, "name": " "}):
            with self.subTest(value=value):
                self.assertIsNone(coerce_gradio_file_path(value))

    def test_supported_file_representations_and_path_precedence(self):
        for value, expected in (
            (" ./audio.wav ", "./audio.wav"),
            ({"path": "audio.wav", "name": "ignored.wav"}, "audio.wav"),
            ({"path": " ", "name": "fallback.wav"}, "fallback.wav"),
            (SimpleNamespace(name="upload.wav"), "upload.wav"),
        ):
            with self.subTest(value=value):
                self.assertEqual(coerce_gradio_file_path(value), expected)


if __name__ == "__main__":
    unittest.main()
