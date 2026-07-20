import unittest

import conftest  # noqa: F401


class TestChatService(unittest.TestCase):
    def test_need_history_context_with_referential(self):
        from services.chat import need_history_context
        self.assertTrue(need_history_context("what about it?", history_exists=True))
        self.assertTrue(need_history_context("look at that", history_exists=True))

    def test_need_history_context_no_referential(self):
        from services.chat import need_history_context
        self.assertFalse(need_history_context("tell me more", history_exists=True))
        self.assertFalse(need_history_context("hello world", history_exists=True))

    def test_need_history_context_no_history(self):
        from services.chat import need_history_context
        self.assertFalse(need_history_context("what about it?", history_exists=False))

    def test_need_history_context_long_query(self):
        from services.chat import need_history_context
        long = "a very long query that exceeds the eight word threshold"
        self.assertFalse(need_history_context(long, history_exists=True))

    def test_load_empty_history(self):
        from services.chat import load_recent_history
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import Base

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        messages = load_recent_history(db, "nonexistent-session")
        db.close()
        self.assertEqual(messages, [])

    def test_save_and_load_history(self):
        from services.chat import load_recent_history
        from repositories.message import create_message
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import Base

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        create_message(db, "session-1", "user", "hello")
        create_message(db, "session-1", "assistant", "hi there")
        messages = load_recent_history(db, "session-1")
        db.close()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "hello")


class TestAuth(unittest.TestCase):
    def test_create_and_authenticate_user(self):
        from repositories.user import create_user, authenticate_user
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import Base

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        user = create_user(db, "authuser", "secret")
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "authuser")

        authed = authenticate_user(db, "authuser", "secret")
        self.assertIsNotNone(authed)
        self.assertEqual(authed.id, user.id)

        failed = authenticate_user(db, "authuser", "wrong")
        self.assertIsNone(failed)
        db.close()


class TestDocumentRepo(unittest.TestCase):
    def test_create_and_list_documents(self):
        from repositories.document import create_user_document, list_user_documents
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import Base

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        doc = create_user_document(
            db, "user1", "doc:1", "test.txt",
            title="Test", doc_type="GENERAL", kind="document",
        )
        self.assertIsNotNone(doc)
        self.assertEqual(doc.filename, "test.txt")

        docs = list_user_documents(db, "user1")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].doc_id, "doc:1")

        empty = list_user_documents(db, "nobody")
        self.assertEqual(empty, [])
        db.close()

    def test_session_crud(self):
        from repositories.message import create_session, list_sessions, get_session_by_id, delete_session
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import Base

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        session = create_session(db, "user1", title="Test Session")
        self.assertIsNotNone(session)
        self.assertEqual(session.title, "Test Session")

        sessions = list_sessions(db, "user1")
        self.assertEqual(len(sessions), 1)

        found = get_session_by_id(db, session.id, "user1")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, session.id)

        deleted = delete_session(db, session.id, "user1")
        self.assertIsNotNone(deleted)

        gone = get_session_by_id(db, session.id, "user1")
        self.assertIsNone(gone)
        db.close()


if __name__ == "__main__":
    unittest.main()
