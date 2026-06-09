from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


def copy_message_session(apps, schema_editor):
    ChatSession = apps.get_model('chatbot', 'ChatSession')
    Conversation = apps.get_model('chatbot', 'Conversation')
    Message = apps.get_model('chatbot', 'Message')

    for message in Message.objects.all():
        if message.conversation_id:
            message.session_id = Conversation.objects.get(pk=message.conversation_id).session_id
            message.save(update_fields=['session_id'])


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0002_rename_chroma_path_chatsession_index_path'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatsession',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='message',
            name='session',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='chatbot.ChatSession'),
        ),
        migrations.RunPython(copy_message_session, reverse_code=migrations.RunPython.noop),
        migrations.RemoveField(
            model_name='message',
            name='conversation',
        ),
        migrations.DeleteModel(
            name='Conversation',
        ),
        migrations.AlterField(
            model_name='message',
            name='session',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbot.ChatSession'),
        ),
    ]
