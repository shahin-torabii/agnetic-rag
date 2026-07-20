import unittest

import conftest  # noqa: F401


class TestGraphState(unittest.TestCase):
    def test_state_dataclass(self):
        from graph.state import AgentState
        from core.types import UserRequest, Intent

        state = AgentState(
            request=UserRequest(query="hello", documents=[], images=[], audio=[]),
            intent=Intent.GENERAL_CHAT,
            user_id="u1",
            session_id="s1",
        )
        self.assertEqual(state.request.query, "hello")
        self.assertEqual(state.intent, Intent.GENERAL_CHAT)

    def test_dispatch_routes_to_image(self):
        from graph.routing import dispatch_branch
        from graph.state import AgentState
        from core.types import UserRequest, Intent

        state = AgentState(
            request=UserRequest(query="find image", documents=[], images=[], audio=[]),
            intent=Intent.IMAGE_SEARCH,
            user_id="u1",
            session_id="s1",
        )
        self.assertEqual(dispatch_branch(state), "image")

    def test_dispatch_routes_to_audio(self):
        from graph.routing import dispatch_branch
        from graph.state import AgentState
        from core.types import UserRequest, Intent

        state = AgentState(
            request=UserRequest(query="transcribe", documents=[], images=[], audio=[]),
            intent=Intent.AUDIO_TRANSCRIBE,
            user_id="u1",
            session_id="s1",
        )
        self.assertEqual(dispatch_branch(state), "audio")

    def test_dispatch_routes_to_general(self):
        from graph.routing import dispatch_branch
        from graph.state import AgentState
        from core.types import UserRequest, Intent

        state = AgentState(
            request=UserRequest(query="hello", documents=[], images=[], audio=[]),
            intent=Intent.GENERAL_CHAT,
            user_id="u1",
            session_id="s1",
        )
        self.assertEqual(dispatch_branch(state), "general")

    def test_dispatch_routes_to_multimodal(self):
        from graph.routing import dispatch_branch
        from graph.state import AgentState
        from core.types import UserRequest, Intent

        state = AgentState(
            request=UserRequest(query="analyze", documents=["doc1"], images=[], audio=[]),
            intent=Intent.DOCUMENT_SUMMARIZE,
            user_id="u1",
            session_id="s1",
        )
        self.assertEqual(dispatch_branch(state), "multimodal")

    def test_pick_k_returns_positive(self):
        from graph.routing import pick_k
        from core.types import Intent

        for intent in Intent:
            k = pick_k(intent)
            self.assertGreater(k, 0, f"pick_k({intent}) returned {k}")


if __name__ == "__main__":
    unittest.main()
