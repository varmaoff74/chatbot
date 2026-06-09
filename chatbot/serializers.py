from rest_framework import serializers
from .models import Message

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "created_at"]


class ChatRequestSerializer(serializers.Serializer):
    session_id = serializers.CharField()
    question = serializers.CharField()