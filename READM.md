# SOBRE A APLICAÇÃO
Esse e um exemplo de agente IA com OLLAMA, uma aplicação sim que recebe pegunta e gerar responstas
A agente está documentada com o swagger.

# Instruções para executar:
Certifique-se de que o Ollama está instalado e rodando no seu sistema (baixe de ollama.ai).
Baixe o modelo especificado: ollama pull llama3.2 (ou altere no .env para outro modelo gratuito disponível).
Ative o ambiente virtual: source venv/bin/activate.
Execute o script: python app.py.


# ARQUIVO DE CONFIGURAÇÃO
.env -  Esse arquivo contem os export e a chave do open api
app.py - arquivo contendo o código de exemplo para executar 

adicionar a chave do open api no .env variavel OPENAI_API_KEY


# Problemas identificados e correções:
Modelo não encontrado: O modelo llama3.2 não estava instalado no Ollama. Executei ollama pull llama3.2 para baixá-lo.
Erros do LangSmith: Mesmo após remover o wrap_openai, o decorador @traceable ainda tentava enviar dados para o LangSmith com uma chave inválida. Desabilitei o tracing no .env definindo LANGSMITH_TRACING=false.



# Criar pasta e ambiente pip
python3 -m venv venv

# Acessar ambiente 
source venv/bin/activate


# Instalar dependencias
pip install -U openai langsmith
pip install -U openai langsmith python-dotenv

# se caso for instalar alguma dependencia adicione no arquivo requirements.txt e rode o comando abaixo, ele vai instalar todos as dependencia contida nele
pip install -r requirements.txt


# RODAR 
python app.py


# URL SWAGGER
http://localhost:5000/apidocs/