import unittest

from src.graph.entity_extractor import SimpleEntityExtractor, normalize_entity


class EntityExtractorTest(unittest.TestCase):
    def test_extracts_capitalized_multiword_entities(self):
        extractor = SimpleEntityExtractor()
        entities = extractor.extract("Ada Lovelace worked on the Analytical Engine with Charles Babbage.")

        self.assertIn("ada lovelace", entities)
        self.assertIn("analytical engine", entities)
        self.assertIn("charles babbage", entities)

    def test_filters_short_and_generic_entities(self):
        extractor = SimpleEntityExtractor()
        entities = extractor.extract("The City of A is in Europe.")

        self.assertNotIn("the", entities)
        self.assertNotIn("a", entities)

    def test_normalize_entity_collapses_whitespace_and_case(self):
        self.assertEqual(normalize_entity("  Ada   Lovelace  "), "ada lovelace")


if __name__ == "__main__":
    unittest.main()
