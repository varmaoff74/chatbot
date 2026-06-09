import os
import sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project2.settings')
import django
django.setup()
from chatbot.models import ChatSession
from django.conf import settings
print('BASE_DIR=', settings.BASE_DIR)
for s in ChatSession.objects.all():
    print('SESSION:', s.session_id)
    print('  doc_path:', s.document_path)
    print('  index_path:', s.index_path)
    print('  doc_exists:', os.path.exists(s.document_path))
    alt_doc_path = os.path.join(settings.BASE_DIR, s.document_path)
    print('  alt_exists:', os.path.exists(alt_doc_path), 'alt_path:', alt_doc_path)
    mp = os.path.join(settings.BASE_DIR, 'media', s.session_id)
    print('  media_dir:', mp, 'exists', os.path.exists(mp))
    print('  chunks:', os.path.exists(os.path.join(mp, 'chunks.pkl')))
    print('  index_direct:', os.path.exists(s.index_path), os.path.exists(os.path.join(settings.BASE_DIR, 'indexes', f'{s.session_id}.faiss')))
    print()