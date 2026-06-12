from query_router import route_query, Intent
from data_gathering import UserRequest


def handle_image(intent, request):
    pass


def handle_audio(intent, request):
    pass


def handle_general(intent, request):
    pass


def handle_document(intent, request):
    pass


def handle_query(request:UserRequest):
    intent = route_query(request)

    match intent:
        case Intent.IMAGE_QA | Intent.IMAGE_EXPLAIN:
            handle_image(intent, request)
        case Intent.AUDIO_QA | Intent.AUDIO_SUMMARIZE | Intent.AUDIO_TRANSCRIBE :
            handle_audio(intent, request)
        case Intent.GENERAL_CHAT | Intent.UNKNOWN:
            handle_general(intent, request)
        case _:
            handle_document(intent, request)
            