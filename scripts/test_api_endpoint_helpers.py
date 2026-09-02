import unittest

import contract_multi_agent_repair as cc_mar
import seeded_multi_agent_planner as seeded


class ApiEndpointHelpersTest(unittest.TestCase):
    def test_versioned_base_is_not_duplicated(self):
        expected = "https://provider.example/v1/chat/completions"
        self.assertEqual(
            seeded.chat_completions_url("https://provider.example/v1"),
            expected,
        )
        self.assertEqual(
            cc_mar.chat_completions_url("https://provider.example/v1"),
            expected,
        )

    def test_endpoint_can_be_passed_directly(self):
        endpoint = "https://provider.example/v1/chat/completions"
        self.assertEqual(seeded.chat_completions_url(endpoint), endpoint)
        self.assertEqual(cc_mar.chat_completions_url(endpoint), endpoint)


if __name__ == "__main__":
    unittest.main()
