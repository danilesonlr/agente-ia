import os
from dotenv import load_dotenv

# 🔧 Carregar .env corretamente independente de onde rodar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

# (opcional) debug — pode remover depois
print("OLLAMA_MODEL:", os.getenv("OLLAMA_MODEL"))

from openai import OpenAI
from langsmith import traceable
from flask import Flask, request, jsonify
from flasgger import Swagger
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Cliente Ollama (usando API compatível com OpenAI)
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
if not ollama_model:
    raise EnvironmentError(
        "OLLAMA_MODEL não encontrada. Defina a variável de ambiente ou adicione OLLAMA_MODEL=<modelo> no arquivo .env."
    )

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

app = Flask(__name__)
swagger = Swagger(app)

# Modelo de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Armazenar o índice FAISS e os documentos
faiss_index = None
documents_chunks = []


def load_pdf_and_create_rag(pdf_path: str):
    """Carrega o PDF e cria o índice FAISS para RAG"""
    global faiss_index, documents_chunks
    
    try:
        # Extrair texto do PDF
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        print(f"✓ PDF carregado com sucesso. Total de caracteres: {len(text)}")
        
        # Dividir o texto em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        documents_chunks = text_splitter.split_text(text)
        print(f"✓ Texto dividido em {len(documents_chunks)} chunks")
        
        # Criar embeddings para todos os chunks
        embeddings = embedding_model.encode(documents_chunks)
        embeddings = np.array(embeddings).astype('float32')
        
        # Criar índice FAISS
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)
        print(f"✓ Índice FAISS criado com sucesso")
        
        return True
    except Exception as e:
        print(f"✗ Erro ao carregar PDF: {e}")
        return False


def retrieve_context(question: str, k: int = 3) -> str:
    """Recupera os chunks mais relevantes para a pergunta"""
    global faiss_index, documents_chunks
    
    if faiss_index is None or len(documents_chunks) == 0:
        return ""
    
    try:
        # Criar embedding da pergunta
        question_embedding = embedding_model.encode([question])
        question_embedding = np.array(question_embedding).astype('float32')
        
        # Buscar os k chunks mais similares
        distances, indices = faiss_index.search(question_embedding, min(k, len(documents_chunks)))
        
        # Recuperar os textos dos chunks
        context_chunks = [documents_chunks[idx] for idx in indices[0]]
        context = "\n\n".join(context_chunks)
        
        return context
    except Exception as e:
        print(f"Erro ao recuperar contexto: {e}")
        return ""


@traceable(name="Chat Pipeline")
def chat_pipeline(question: str):
    # Recuperar contexto do PDF
    context = retrieve_context(question)
    
    # Construir mensagens
    if context:
        system_content = "Você é um assistente útil. Responda baseado no contexto fornecido abaixo:\n\n" + context
    else:
        system_content = "Você é um assistente útil."
    
    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )

    return response.choices[0].message.content


@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint para fazer perguntas ao agente RAG com Ollama
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              description: A pergunta a ser feita ao agente baseado no PDF
              example: "Quais são as dicas de saúde para mulheres?"
    responses:
      200:
        description: Resposta do agente baseada no contexto do PDF
        schema:
          type: object
          properties:
            response:
              type: string
              example: "Resposta gerada pelo modelo Ollama baseada no PDF."
      400:
        description: Erro se a pergunta não for fornecida
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Question is required"
    """
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    response = chat_pipeline(question)
    return jsonify({'response': response})


if __name__ == "__main__":
    # Carregar PDF na inicialização
    pdf_path = os.path.join(BASE_DIR, "saudedamulher.pdf")
    if os.path.exists(pdf_path):
        print(f"📄 Carregando PDF: {pdf_path}")
        load_pdf_and_create_rag(pdf_path)
    else:
        print(f"⚠ PDF não encontrado em: {pdf_path}")
    
    print("\n🚀 Iniciando servidor Flask...")
    app.run(debug=True)