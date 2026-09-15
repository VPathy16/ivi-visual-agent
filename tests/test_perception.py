import io
import unittest

from PIL import Image

from ivi_agent.perception import (
    extract_screen_titles,
    extract_ui_elements,
    extract_visible_text,
    hash_distance,
    parse_ocr_tsv,
    perceptual_hash,
    prepare_model_image,
    prepare_grid_grounding_image,
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

    def test_preserves_toggle_state_for_safe_planning(self) -> None:
        source = """<hierarchy><node bounds='[0,0][100,100]'>
          <node text='Bluetooth' class='android.widget.Switch' checkable='true'
                checked='true' clickable='true' bounds='[10,10][90,40]' />
        </node></hierarchy>"""
        element = extract_ui_elements(source)[0]
        self.assertTrue(element.checkable)
        self.assertTrue(element.checked)
        self.assertTrue(element.stateful)

    def test_recognizes_accessibility_on_off_control_as_stateful(self) -> None:
        source = """<hierarchy><node bounds='[0,0][100,100]'>
          <node content-desc='Off, Bluetooth, Button' class='android.widget.LinearLayout'
                clickable='true' bounds='[10,10][90,40]' />
        </node></hierarchy>"""
        self.assertTrue(extract_ui_elements(source)[0].stateful)

    def test_extracts_only_semantically_identified_screen_titles(self) -> None:
        source = """<hierarchy><node bounds='[0,0][100,100]'>
          <node text='Bluetooth' resource-id='ivi:id/toolbar_title'
                class='android.widget.TextView' bounds='[10,0][90,20]' />
          <node text='Notification settings' resource-id='android:id/title'
                class='android.widget.TextView' bounds='[10,70][90,90]' />
        </node></hierarchy>"""
        self.assertEqual(extract_screen_titles(source), ["Bluetooth"])

    def test_extracts_prefixed_ivi_toolbar_titles(self) -> None:
        source = """<hierarchy><node bounds='[0,0][100,100]'>
          <node text='Settings' resource-id='car.ui:id/car_ui_toolbar_title'
                class='android.widget.TextView' bounds='[10,0][90,20]' />
        </node></hierarchy>"""
        self.assertEqual(extract_screen_titles(source), ["Settings"])

    def test_extracts_visible_text_without_requiring_clickability(self) -> None:
        source = """<hierarchy><node bounds='[0,0][100,100]'>
          <node text='Connected devices' content-desc='Bluetooth page'
                clickable='false' bounds='[10,0][90,20]' />
        </node></hierarchy>"""
        self.assertEqual(
            extract_visible_text(source), ["Connected devices", "Bluetooth page"]
        )

    def test_resizes_large_image_for_model(self) -> None:
        source = io.BytesIO()
        Image.new("RGB", (2000, 1000), "blue").save(source, format="PNG")
        resized = prepare_model_image(source.getvalue(), 500)
        with Image.open(io.BytesIO(resized)) as result:
            self.assertEqual(result.size, (500, 250))

    def test_grid_grounding_image_preserves_aspect_ratio(self) -> None:
        source = io.BytesIO()
        Image.new("RGB", (1200, 600), "black").save(source, format="PNG")
        grounded = prepare_grid_grounding_image(source.getvalue(), 600)
        with Image.open(io.BytesIO(grounded)) as result:
            self.assertEqual(result.size, (600, 300))

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
