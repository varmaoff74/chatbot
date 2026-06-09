import os
import time
import uuid
import faiss
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import ChatSession, Message
from .serializers import ChatRequestSerializer
from rag.loader import load_pdf
from rag.chunker import recursive_chunking, save_chunks, load_chunks
from rag.pipeline import query_pipeline
from rag.embeddings import get_embedding


def chat_ui(request):
    return render(request, "frontend.html")


class SessionListAPIView(APIView):

    def get(self, request):

        sessions = ChatSession.objects.order_by("-id")

        return Response({
            "sessions": [
                {
                    "session_id": s.session_id,
                    "document_name": s.document_name,
                    "created_at": s.created_at
                }
                for s in sessions
            ]
        })

class HistoryList(APIView):

    def get(self, request, session_id=None):
        if session_id:
            chat_session = ChatSession.objects.filter(session_id=session_id).first()
            if not chat_session:
                return Response({"error": "Invalid session_id. Please upload a PDF first or select a valid session."}, status=status.HTTP_400_BAD_REQUEST)

            messages = Message.objects.filter(
                session=chat_session
            ).order_by("created_at")

            return Response({
                "session_id": session_id,
                "document_name": chat_session.document_name,
                "document_names": [n.strip() for n in chat_session.document_name.split("|") if n.strip()],
                "history": [
                    {
                        "role": m.role,
                        "content": m.content
                    }
                    for m in messages
                ]
            })

        sessions = ChatSession.objects.order_by("-created_at")
        return Response({
            "sessions": [
                {
                    "session_id": s.session_id,
                    "document_name": s.document_name,
                    "document_names": [n.strip() for n in s.document_name.split("|") if n.strip()],
                    "history": [
                        {
                            "role": m.role,
                            "content": m.content
                        }
            for m in Message.objects.filter(session=s).order_by("created_at")] } for s in sessions]})


class UploadView(APIView):

    def post(self, request):
        uploaded_file = request.FILES.get("file")
        session_id = request.POST.get("session_id")

        if not uploaded_file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        session = None
        if session_id:
            session = ChatSession.objects.filter(session_id=session_id).first()

        if not session:
            session_id = str(uuid.uuid4())      

        media_root = os.path.join(settings.BASE_DIR, "media")
        base_dir = os.path.join(media_root, session_id)
        os.makedirs(base_dir, exist_ok=True)

        upload_start = time.perf_counter()
        fs = FileSystemStorage(location=base_dir)
        file_name = fs.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join(base_dir, file_name)
        upload_time = time.perf_counter() - upload_start

        load_start = time.perf_counter()
        docs = load_pdf(base_dir)
        load_pdf_time = time.perf_counter() - load_start

        chunk_start = time.perf_counter()
        chunks = recursive_chunking(docs)
        chunking_time = time.perf_counter() - chunk_start

        embed_start = time.perf_counter()
        embeddings = get_embedding(chunks, batch_size=int(os.environ.get("EMBED_BATCH_SIZE", 128)))
        embedding_time = time.perf_counter() - embed_start
        if embeddings is None or (hasattr(embeddings, '__len__') and len(embeddings) == 0):
            return Response({"error": "Unable to extract text from the uploaded document."}, status=status.HTTP_400_BAD_REQUEST)

        dim = len(embeddings[0])
        index_root = os.path.join(settings.BASE_DIR, "indexes")
        os.makedirs(index_root, exist_ok=True)
        index_path = os.path.join(index_root, f"{session_id}.faiss")

        existing_index_path = None
        if session:
            if session.index_path and os.path.exists(session.index_path):
                existing_index_path = session.index_path
            elif os.path.exists(index_path):
                existing_index_path = index_path

            # Only reuse an existing index if the session's previous documents are still available.
            document_folder_exists = False
            if session.document_path:
                if os.path.exists(session.document_path):
                    document_folder_exists = True
                else:
                    alt_doc_path = os.path.join(settings.BASE_DIR, session.document_path)
                    if os.path.exists(alt_doc_path):
                        document_folder_exists = True

            if not document_folder_exists:
                existing_index_path = None

        index_start = time.perf_counter()
        if existing_index_path:
            index = faiss.read_index(existing_index_path)
            index_path = existing_index_path
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(index, index_path)
        index_time = time.perf_counter() - index_start

        save_chunks_start = time.perf_counter()
        chunk_file_path = os.path.join(base_dir, "chunks.pkl")
        save_chunks(chunks, chunk_file_path)
        save_chunks_time = time.perf_counter() - save_chunks_start

        db_start = time.perf_counter()
        if session:
            session.document_name = f"{session.document_name} | {file_name}"
            session.document_path = base_dir
            session.save(update_fields=["document_name", "document_path"])
        else:
            session = ChatSession.objects.create(
                session_id=session_id,
                document_name=file_name,
                document_path=base_dir,
                index_path=index_path
            )
        db_time = time.perf_counter() - db_start

        return Response({
            "session_id": session.session_id,
            "message": "PDF uploaded and indexed",
            "document_names": [n.strip() for n in session.document_name.split("|") if n.strip()],
            "chunks": len(chunks),
            "timings": {
                "upload_save_sec": round(upload_time, 4),
                "pdf_load_sec": round(load_pdf_time, 4),
                "chunking_sec": round(chunking_time, 4),
                "embedding_sec": round(embedding_time, 4),
                "index_write_sec": round(index_time, 4),
                "chunk_save_sec": round(save_chunks_time, 4),
                "db_save_sec": round(db_time, 4),
            }
        })

class ChatAPIView(APIView):

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data["session_id"]
        question = serializer.validated_data["question"]

        chat_session = ChatSession.objects.filter(session_id=session_id).first()
        if not chat_session:
            return Response({"error": "Invalid session_id. Please upload a PDF first or select an existing session."}, status=status.HTTP_400_BAD_REQUEST)

        index_root = os.path.join(settings.BASE_DIR, "indexes")
        index_path = chat_session.index_path or os.path.join(index_root, f"{session_id}.faiss")
        if not os.path.exists(index_path):
            return Response({"error": "This session has no saved index. Upload a PDF to start a new session."}, status=status.HTTP_400_BAD_REQUEST)
        if index_path != chat_session.index_path:
            chat_session.index_path = index_path
            chat_session.save(update_fields=["index_path"])

        document_path = chat_session.document_path
        if document_path and not os.path.exists(document_path):
            alt = os.path.join(settings.BASE_DIR, document_path)
            if os.path.exists(alt):
                document_path = alt
            else:
                document_path = None

        media_dir = os.path.join(settings.BASE_DIR, "media", session_id)
        if not document_path and os.path.exists(media_dir):
            document_path = media_dir

        if not document_path:
            return Response({"error": "This session has no uploaded document available. Upload a PDF to continue or select another session."}, status=status.HTTP_400_BAD_REQUEST)
        if document_path != chat_session.document_path:
            chat_session.document_path = document_path
            chat_session.save(update_fields=["document_path"])

        index = faiss.read_index(index_path)

        chunks_file = os.path.join(document_path, "chunks.pkl") if os.path.isdir(document_path) else os.path.join(os.path.dirname(document_path), "chunks.pkl")
        if os.path.exists(chunks_file):
            chunks = load_chunks(chunks_file)
        else:
            docs = load_pdf(document_path)
            if not docs:
                return Response({"error": "No document content found for this session. Upload a valid PDF file."}, status=status.HTTP_400_BAD_REQUEST)
            chunks = recursive_chunking(docs)
            save_chunks(chunks, chunks_file)

        if not chunks:
            return Response({"error": "Unable to extract text chunks from the uploaded document. Please upload a valid PDF."}, status=status.HTTP_400_BAD_REQUEST)

        history = Message.objects.filter(session=chat_session).order_by("created_at")
        history_lines = [f"{m.role}: {m.content}" for m in history]
        # include session document names in the conversation history so the generator
        # can answer meta-questions like "List all the documents uploaded"
        doc_names = [n.strip() for n in chat_session.document_name.split("|") if n.strip()]
        if doc_names:
            history_lines.insert(0, "Session documents: " + ", ".join(doc_names))
        history_text = "\n".join(history_lines)
                                                                                                                                                                                                                                                                                                                                                                                                                                         
        answer = query_pipeline(
            index,
            question,
            chunks,
            history_text=history_text,
            vector_k=20,
            bm25_k=20,
            fused_k=10,
            top_k=3,
        )
        answer = " ".join(answer.splitlines()).strip()

        Message.objects.bulk_create([
            Message(session=chat_session, role="user", content=question),
            Message(session=chat_session, role="assistant", content=answer),
        ])

        return Response({
            "answer": answer,
            "session_id": chat_session.session_id,
        })

