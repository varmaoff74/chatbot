from django.urls import path
from .views import ChatAPIView, SessionListAPIView, HistoryList, UploadView, chat_ui


urlpatterns = [
    path("", chat_ui, name='chat_ui'),
    path("chat/", ChatAPIView.as_view()),
    path("sessions/", SessionListAPIView.as_view()),
    path("messages/", HistoryList.as_view()),
    path("messages/<str:session_id>/", HistoryList.as_view()),
    path("upload/", UploadView.as_view()),
]