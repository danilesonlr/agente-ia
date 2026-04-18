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

# Cliente Ollama (usando API compatível com OpenAI)
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
if not ollama_model:
    raise EnvironmentError(
        "OLLAMA_MODEL não encontrada. Defina a variável de ambiente ou adicione OLLAMA_MODEL=<modelo> no arquivo .env."
    )

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

app = Flask(__name__)
swagger = Swagger(app)


@traceable(name="Chat Pipeline")
def chat_pipeline(question: str):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
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
              description: A pergunta a ser feita ao agente
              example: "Can you summarize this morning's meetings?"
    responses:
      200:
        description: Resposta do agente baseada no contexto
        schema:
          type: object
          properties:
            response:
              type: string
              example: "Resposta gerada pelo modelo Ollama."
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
    app.run(debug=True)