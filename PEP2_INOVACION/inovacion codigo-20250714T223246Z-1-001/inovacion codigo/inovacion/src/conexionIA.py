# app.py
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
# Variables para almacenar información entre llamadas
user_question = None
user_gravity = None
user_name = None
user_lastname = None
user_rut = None


# Configuración inicial
app = Flask(__name__)
CORS(app)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Modelos y base de datos vectorial
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)  # Dimensión del modelo de embeddings
documents = []  # Almacenamiento temporal de documentos

# Configuración de Gemini
generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)
# Función para imprimir información del usuario
def imprimir_info_usuario():
    print("\n--- Información del Usuario ---")
    print(f"Pregunta: {user_question}")
    print(f"Gravedad estimada: {user_gravity}")
    print(f"Nombre: {user_name}")
    print(f"Apellido: {user_lastname}")
    print(f"RUT: {user_rut}")
    print("--------------------------------\n")

# Función para inicializar datos de ejemplo
def initialize_sample_data():
    sample_docs = [
        "Si en la queja/reclamo tiene que ver con: demanda, robo, muerte, perdida del pedido, la gravedad es maxima",
        "Si en la queja/reclamo tiene que ver con: problemas de pagos, problemas de reparto, problemas con la empresa empleadora, la gravedad es alta",
        "Si en la queja/reclamo tiene que ver con: cambio de vehiculo, incomodidad con la ruta, la gravedad es media",
        "los que no consideres que entren en las categorias anteriores debes evaluarlos por tu cuente, por favor ademas de una respuesta al usuario da un valor numerico asociado a la gravedad"
    ]
    
    embeddings = embedding_model.encode(sample_docs)
    index.add(np.array(embeddings))
    documents.extend(sample_docs)

# Funciones principales
@lru_cache(maxsize=1024)
def text_to_vector(question):
    return embedding_model.encode([question])[0]

def semantic_search(query_vector, k=3):
    distances, indices = index.search(np.array([query_vector]), k)
    return indices[0].tolist()

def build_prompt(question, context):
    return f"""
    Eres un trabajador de Fltzy, la cual es una empresa sobre gestion de repartidores, 
    tu trabajo es responder a los usuarios, ademas debes pedirle su nombre, apellido y su rut

    Contexto:
    {''.join([f'- {doc}\n' for doc in context])}

    Pregunta: {question}
    Respuesta: 
    """


def build_prompt2(question, context):
    return f"""
    Eres un trabajador de Fltzy, la cual es una empresa sobre gestion de repartidores, 
    tu trabajo es dar un valor de gravedad de la consulta basandote solo en el contexto, da un valor del 1 al 100, siendo el 100 lo mas alto,
    solo responde con ese valor numerico

    Contexto:
    {''.join([f'- {doc}\n' for doc in context])}

    Pregunta: {question}
    Respuesta: 
    """

def build_segunda_interaccion(question, context):
    return f"""
    Eres un trabajador de Fltzy, la cual es una empresa sobre gestion de repartidores, 
    tu trabajo es responder a los usuarios y ademas debes obtener el nombre, apellido y rut del usuario
    debes pedir que se envien en formato nombre: <nombre>, apellido: <apellido>, rut: <rut>

    Contexto:
    {''.join([f'- {doc}\n' for doc in context])}

    Pregunta: {question}
    Respuesta: 
    """

def build_tercera_interaccion(question, context):
    return f"""
    Eres un trabajador de Fltzy, la cual es una empresa sobre gestion de repartidores, 
    tu trabajo es responder a los usuarios tras obtener el nombre, apellido y rut del usuario
    debes pedir que te den más información sobre el incidente ocurrido

    Contexto:
    {''.join([f'- {doc}\n' for doc in context])}

    Pregunta: {question}
    Respuesta: 
    """

@app.route('/ask', methods=['POST'])
def ask():
    global user_question, user_gravity  # declarar como globales para modificarlas

    try:
        data = request.json
        question = data['question']
        user_question = question  # Guardamos la pregunta

        query_vector = text_to_vector(question)
        context_indices = semantic_search(query_vector)
        context = [documents[i] for i in context_indices]

        prompt = build_prompt(question, context)
        response = model.generate_content(prompt)

        prompt2 = build_prompt2(question, context)
        response2 = model.generate_content(prompt2)

        user_gravity = response2.text.strip()  # Guardamos la gravedad

        return jsonify({
            "question": user_question,
            "answer": response.text,
            "context_sources": context_indices,
            "gravedad": user_gravity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask2', methods=['POST'])
def ask2():
    global user_name, user_lastname, user_rut, user_context, user_new_gravity  # declarar como globales

    try:
        data = request.json
        question = data['question']

        query_vector = text_to_vector(question)
        context_indices = semantic_search(query_vector)
        context = [documents[i] for i in context_indices]

        prompt = build_segunda_interaccion(question, context)
        response = model.generate_content(prompt)
        credenciales = response.text.strip()

        # (OPCIONAL) Lógica simple para extraer credenciales — puedes ajustar con regex si quieres mayor precisión.
        lines = credenciales.split('\n')
        for line in lines:
            if "nombre" in line.lower():
                user_name = line.split(":")[-1].strip()
            elif "apellido" in line.lower():
                user_lastname = line.split(":")[-1].strip()
            elif "rut" in line.lower():
                user_rut = line.split(":")[-1].strip()

        prompt3 = build_tercera_interaccion(question, context)
        response3 = model.generate_content(prompt3)
        user_context = response3.text.strip()  # Guardamos el contexto

        user_context = data['user_context']  # Este es el nuevo contexto detallado del usuario

        # Simulamos un contexto similar al de la búsqueda semántica
        context = [user_context]

        # Usamos el mismo prompt que da solo el número de gravedad
        prompt2 = build_prompt2(user_question, context)
        user_new_gravity = model.generate_content(prompt2)

        user_gravity = response2.text.strip()  # Guardamos la gravedad
        return jsonify({
            "question": question,
            "credenciales": credenciales,
            "context_sources": context_indices,
            "name": user_name,
            "lastname": user_lastname,
            "rut": user_rut,
            "gravedad": user_new_gravity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    


# Interfaz de prueba simple
@app.route('/test', methods=['GET'])
def test_interface():
    return render_template_string('''
        <form action="/ask" method="post" onsubmit="event.preventDefault(); fetchAsk()">
            <textarea id="question" rows="4" cols="50"></textarea><br>
            <button type="submit">Enviar</button>
        </form>
        <div id="result"></div>
        <script>
            function fetchAsk() {
                const question = document.getElementById('question').value;
                fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `<strong>Pregunta:</strong> ${data.question}<br>
                                          <strong>Respuesta:</strong> ${data.answer}<br>
                                          <strong>Contexto usado:</strong> ${data.context_sources}`;
                });
            }
        </script>
    ''')



@app.route('/info', methods=['GET'])
def info_usuario():
    try:
        imprimir_info_usuario()
        return jsonify({
            "pregunta": user_question,
            "gravedad": user_gravity,
            "nombre": user_name,
            "apellido": user_lastname,
            "rut": user_rut
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    initialize_sample_data()
    app.run(debug=True)