from django.db import models

class ChatSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    document_name = models.CharField(max_length=255)
    document_path = models.TextField()
    index_path = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class Message(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    role = models.CharField(max_length=20)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)