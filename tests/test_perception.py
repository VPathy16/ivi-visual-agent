import io
import unittest

from PIL import Image

from ivi_agent.perception import (
    extract_ui_elements,
    hash_distance,
    parse_ocr_tsv,
    perceptual_hash,
    prepare_model_image,
)


class PerceptionTests(unittest.TestCase):
    def test_extracts_clickable_candidate_with_normalized_center(self) -> None:
        source = """<hierarchy><node bounds='[0,0][1000,500]'>
          <node text='Bluetooth' class='android.widget.Button' package='vehicle.settings'
                resource-id='vehicle:id/bluetooth' clickable='true'
                bounds='[600,100][900,200]' />
        </node></hierarchy>"""
        elements = extract_ui_elements(source)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].label, "Bluetooth")
        self.assertEqual(elements[0].center, (0.75, 0.3))

    def test_uses_child_text_for_clickable_parent(self) -> None:
        source = """<hierarchy><node bounds='[0,0][100,100]'>
          <node clickable='true' bounds='[10,10][90,40]'>
            <node text='Settings' bounds='[20,10][80,40]' />
          </node>
        </node></hierarchy>"""
        self.assertEqual(extract_ui_elements(source)[0].label, "Settings")

    def test_resizes_large_image_for_model(self) -> None:
        source = io.BytesIO()
        Image.new("RGB", (2000, 1000), "blue").save(source, format="PNG")
        resized = prepare_model_image(source.getvalue(), 500)
        with Image.open(io.BytesIO(resized)) as result:
            self.assertEqual(result.size, (500, 250))

    def test_perceptual_hash_ignores_identical_content(self) -> None:
        source = io.BytesIO()
        Image.new("RGB", (30, 30), "white").save(source, format="PNG")
        first = perceptual_hash(source.getvalue())
        second = perceptual_hash(source.getvalue())
        self.assertEqual(hash_distance(first, second), 0)

    def test_parses_ocr_candidate(self) -> None:
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t100\t50\t200\t40\t95.0\tBluetooth\n"
        )
        elements = parse_ocr_tsv(tsv, 1000, 500, start_id=4)
        self.assertEqual(elements[0].id, 4)
        self.assertEqual(elements[0].label, "Bluetooth")
        self.assertEqual(elements[0].source, "ocr")
        self.assertEqual(elements[0].center, (0.2, 0.14))


if __name__ == "__main__":
    unittest.main()
