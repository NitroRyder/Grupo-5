from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
import re
from functools import lru_cache
import requests

import google.generativeai as genai
from dotenv import load_dotenv


import os
import database as db

template_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
template_dir = os.path.join(template_dir, 'src', 'templates')

app = Flask(__name__, template_folder = template_dir) 
CORS(app)

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])




# Función para obtener el user_id desde el RUT
def obtener_user_id(rut):
    cursor = cursor = db.get_cursor()  
    cursor.execute("SELECT id FROM users WHERE rut = %s", (rut,))
    result = cursor.fetchone()
    cursor.close()  # <-- MUY IMPORTANTE
    if result:
        return result[0]
    else:
        raise Exception("Usuario no encontrado con RUT: " + rut)


# (Opcional) función para clasificar nivel según gravedad
def determinar_nivel(gravedad):
    try:
        gravedad = float(gravedad)  # Asegura que sea número aunque venga como string

        if gravedad >= 80:
            return "alto"
        elif gravedad >= 50:
            return "medio"
        elif gravedad >= 0:
            return "bajo"
        else:
            return "desconocido"
    except ValueError:
        return "desconocido"


# Ruta para mostrar usuarios (HTML)
@app.route('/')
def home():
    cursor = cursor = db.get_cursor()  
    cursor.execute("SELECT * FROM users")
    myresult = cursor.fetchall()
    insertObject = []
    columnNames = [column[0] for column in cursor.description]
    for record in myresult:
        insertObject.append(dict(zip(columnNames, record)))
    cursor.close()
    return render_template('index.html', data=insertObject)

# --- RUTAS API PARA OBTENER DATOS EN JSON ---

@app.route('/api/users')
def api_users():
    cursor = cursor = db.get_cursor()  
    cursor.execute("SELECT * FROM users")
    myresult = cursor.fetchall()
    columnNames = [column[0] for column in cursor.description]
    users = [dict(zip(columnNames, record)) for record in myresult]
    cursor.close()
    return jsonify(users)

@app.route('/api/questions')
def api_questions():
    cursor = cursor = db.get_cursor()  
    cursor.execute("SELECT * FROM questions")
    myresult = cursor.fetchall()
    columnNames = [column[0] for column in cursor.description]
    questions = [dict(zip(columnNames, record)) for record in myresult]
    cursor.close()
    return jsonify(questions)

@app.route('/api/answers')
def api_answers():
    cursor = cursor = db.get_cursor()  
    cursor.execute("SELECT * FROM answers")
    myresult = cursor.fetchall()
    columnNames = [column[0] for column in cursor.description]
    answers = [dict(zip(columnNames, record)) for record in myresult]
    cursor.close()
    return jsonify(answers)

@app.route('/api/question_answer')
def api_question_answer():
    cursor = cursor = db.get_cursor()  
    cursor.execute("SELECT * FROM question_answer")
    myresult = cursor.fetchall()
    columnNames = [column[0] for column in cursor.description]
    qa = [dict(zip(columnNames, record)) for record in myresult]
    cursor.close()
    return jsonify(qa)

# --------------------------------------------

# Ruta para guardar usuarios
@app.route('/user', methods=['POST'])
def addUser():
    rut = request.form['rut']
    name = request.form['name']
    lastname = request.form['lastname']

    if rut and name and lastname:
        cursor = cursor = db.get_cursor()  
        sql = "INSERT INTO users (rut, name, lastname) VALUES (%s, %s, %s)"
        data = (rut, name, lastname)
        cursor.execute(sql, data)
        db.database.commit()
    return redirect(url_for('home'))

# Ruta para agregar preguntas
@app.route('/question', methods=['POST'])
def addQuestion():
    user_id = request.form['user_id']
    question_text = request.form['question_text']

    if user_id and question_text:
        cursor = cursor = db.get_cursor()  
        sql = "INSERT INTO questions (user_id, question_text) VALUES (%s, %s)"
        data = (user_id, question_text)
        cursor.execute(sql, data)
        db.database.commit()
    return redirect(url_for('home'))

# Ruta para obtener preguntas de un usuario
@app.route('/questions/<int:user_id>')
def getQuestions(user_id):
    cursor = cursor = db.get_cursor()  
    sql = "SELECT question_text FROM questions WHERE user_id = %s"
    cursor.execute(sql, (user_id,))
    questions = cursor.fetchall()
    cursor.close()
    return render_template('questions.html', questions=questions, user_id=user_id)

# Ruta para agregar respuestas
@app.route('/answer', methods=['POST'])
def addAnswer():
    user_id = request.form['user_id']
    answer_text = request.form['answer_text']

    if user_id and answer_text:
        cursor = cursor = db.get_cursor()  
        sql = "INSERT INTO answers (user_id, answer_text) VALUES (%s, %s)"
        data = (user_id, answer_text)
        cursor.execute(sql, data)
        db.database.commit()
    return redirect(url_for('home'))

# Ruta para relacionar preguntas y respuestas (ahora incluye gravedad y nivel)
@app.route('/question_answer', methods=['POST'])
def linkQuestionAnswer():
    question_id = request.form['question_id']
    answer_id = request.form['answer_id']
    gravedad = request.form['gravedad']
    nivel = request.form['nivel']

    if question_id and answer_id and gravedad and nivel:
        cursor = cursor = db.get_cursor()  
        sql = "INSERT INTO question_answer (question_id, answer_id, gravedad, nivel) VALUES (%s, %s, %s, %s)"
        data = (question_id, answer_id, gravedad, nivel)
        cursor.execute(sql, data)
        db.database.commit()
    return redirect(url_for('home'))

# Ruta para eliminar usuarios
@app.route('/delete/<int:id>')
def delete(id):
    cursor = cursor = db.get_cursor()  

    # Eliminar relaciones en la tabla question_answer
    sql = "DELETE qa FROM question_answer qa INNER JOIN questions q ON qa.question_id = q.id WHERE q.user_id = %s"
    cursor.execute(sql, (id,))

    # Eliminar respuestas asociadas al usuario
    sql = "DELETE FROM answers WHERE user_id = %s"
    cursor.execute(sql, (id,))

    # Eliminar preguntas asociadas al usuario
    sql = "DELETE FROM questions WHERE user_id = %s"
    cursor.execute(sql, (id,))

    # Eliminar el usuario
    sql = "DELETE FROM users WHERE id = %s"
    cursor.execute(sql, (id,))

    db.database.commit()
    cursor.close()
    return redirect(url_for('home'))

# Ruta para editar usuarios
@app.route('/edit/<int:id>', methods=['POST'])
def edit(id):
    rut = request.form['rut']
    name = request.form['name']
    lastname = request.form['lastname']

    if rut and name and lastname:
        cursor = cursor = db.get_cursor()  
        sql = "UPDATE users SET rut = %s, name = %s, lastname = %s WHERE id = %s"
        data = (rut, name, lastname, id)
        cursor.execute(sql, data)
        db.database.commit()
    return redirect(url_for('home'))


# CODIGO IA
CORS(app)

# Modelo de embeddings y base de datos vectorial
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
documents = []
tipos_y_prioridades = []

# Almacén temporal de datos del usuario
user_data = {
    "nombre": None,
    "apellido": None,
    "rut": None,
    "pregunta_original": None,
}

def initialize_data_from_json():
    global documents, tipos_y_prioridades
    with open("../data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Cargando {len(data)} entradas del JSON.")
        for entry in data:
            tipo = entry["tipo"]
            prioridad = entry["prioridad"]
            for contexto in entry["contexto"]:
                documents.append(contexto)
                tipos_y_prioridades.append({"tipo": tipo, "prioridad": prioridad})
        print(f"Total documentos cargados: {len(documents)}")
        embeddings = embedding_model.encode(documents)
        index.add(np.array(embeddings))
        print(f"Índice FAISS inicializado con {index.ntotal} vectores.")


@lru_cache(maxsize=1024)
def text_to_vector(text):
    return embedding_model.encode([text])[0]

def semantic_search(query_vector, k=3):
    distances, indices = index.search(np.array([query_vector]), k)
    # Filtrar índices válidos (>=0)
    valid_indices = [i for i in indices[0].tolist() if i >= 0]
    print(f"semantic_search - Distancias: {distances[0].tolist()}")
    print(f"semantic_search - Índices filtrados válidos: {valid_indices}")
    return valid_indices


def build_prompt(question, context):
    return f"""
Eres un trabajador de Fltzy, una empresa de gestión de repartidores.
Tu trabajo es responder educadamente a los reclamos de los usuarios.
Además, DEBES INDICARLES QUE SI QUIEREN AÑADIR ALGUNA INFORMACION MAS PUEDEN HACERLO.

Contexto relacionado:
{''.join([f'- {doc}\n' for doc in context])}

Pregunta del usuario: {question}
Respuesta:
"""

def build_prompt_gravedad(question, context):
    return f"""
Eres un trabajador de Fltzy, una empresa de gestión de repartidores.
Analiza la siguiente pregunta basándote en el contexto entregado y entrega un valor de gravedad entre 1 y 100.

Contexto relacionado:
{''.join([f'- {doc}\n' for doc in context])}

Pregunta del usuario: {question}
Solo responde con un valor numérico:
"""
def ask_gemini(prompt, model_name="models/gemini-1.5-pro-latest"):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Gemini API error: {e}")


@app.route('/ask', methods=['POST'])
def ask():
    try:
        print("=== /ask request received ===")
        data = request.json
        print(f"Request JSON data: {data}")

        question = data.get('question')
        if not question:
            return jsonify({"error": "Falta el campo 'question' en la petición."}), 400
        
        # Guardar pregunta original en user_data
        user_data["pregunta_original"] = question

        print(f"Pregunta recibida: {question}")

        query_vector = text_to_vector(question)
        print(f"Vector calculado para la pregunta: {query_vector[:5]}... (primeros 5 valores)")

        context_indices = semantic_search(query_vector)
        print(f"Índices de contexto obtenidos: {context_indices}")

        if not context_indices:
            print("No se encontró contexto relevante para la pregunta.")
            return jsonify({
                "question": question,
                "answer": "No encontré información relevante para responder a tu pregunta.",
                "context_sources": [],
                "gravedad": None,
                "nivel": None
            }), 200

        context = [documents[i] for i in context_indices]
        print(f"Contexto recuperado ({len(context)} documentos): {context}")

        prompt = build_prompt(question, context)
        prompt2 = build_prompt_gravedad(question, context)

        answer = ask_gemini(prompt)
        gravedad = ask_gemini(prompt2)

        # Extraer número desde el string
        match = re.search(r'\d+', gravedad)
        if match:
            nivel = int(match.group())
            print(f"Número de gravedad extraído: {nivel}")
        else:
            nivel = 0  # valor por defecto si no se encontró número

        GRAVEDAD = determinar_nivel(nivel)

        rut = user_data.get('rut')
        if not rut:
            print("No se encontró RUT en user_data.")
            return jsonify({"error": "Falta el RUT del usuario para continuar."}), 400

        user_id = obtener_user_id(rut)

        # === Guardar pregunta ===
        with db.get_cursor() as cursor1:
            sql_question = "INSERT INTO questions (user_id, question_text) VALUES (%s, %s)"
            cursor1.execute(sql_question, (user_id, question))
            db.database.commit()
            question_id = cursor1.lastrowid

        # === Guardar respuesta ===
        with db.get_cursor() as cursor2:
            sql_answer = "INSERT INTO answers (user_id, answer_text) VALUES (%s, %s)"
            cursor2.execute(sql_answer, (user_id, answer))
            db.database.commit()
            answer_id = cursor2.lastrowid

        # === Guardar relación pregunta-respuesta con gravedad ===
        with db.get_cursor() as cursor3:
            sql_link = "INSERT INTO question_answer (question_id, answer_id, gravedad, nivel) VALUES (%s, %s, %s, %s)"
            cursor3.execute(sql_link, (question_id, answer_id, GRAVEDAD, nivel))
            db.database.commit()

        return jsonify({
            "question": question,
            "answer": answer,
            "context_sources": context,
            "gravedad": GRAVEDAD,
            "nivel": nivel
        })

    except Exception as e:
        print(f"Exception caught in /ask: {e}")
        return jsonify({"error": str(e)}), 500
    
# ======== ENDPOINT PARA CREDENCIALES ========
@app.route('/credenciales', methods=['POST'])
def recibir_credenciales():
    try:
        data = request.json
        entrada = data.get('texto', '')

        nombre = apellido = rut = None

        nombre_match = re.search(r'nombre\s*:\s*(\w+)', entrada, re.IGNORECASE)
        apellido_match = re.search(r'apellido\s*:\s*(\w+)', entrada, re.IGNORECASE)
        rut_match = re.search(r'rut\s*:\s*([\d\.\-kK]+)', entrada, re.IGNORECASE)

        if nombre_match: nombre = nombre_match.group(1)
        if apellido_match: apellido = apellido_match.group(1)
        if rut_match: rut = rut_match.group(1)

        if not (nombre and apellido and rut):
            partes = re.split(r'\s+|\n+', entrada.strip())
            partes = [p for p in partes if p]
            if len(partes) >= 3:
                nombre = nombre or partes[0]
                apellido = apellido or partes[1]
                rut_candidates = [p for p in partes[2:] if re.match(r'[\d\.\-kK]+', p)]
                rut = rut or (rut_candidates[0] if rut_candidates else None)

        if not (nombre and apellido and rut):
            return jsonify({
                "error": "No se pudieron extraer correctamente nombre, apellido y RUT.",
                "mensaje": "Envíe las credenciales como:\n- nombre: Juan\n- apellido: Pérez\n- rut: 12.345.678-9"
            }), 400

        user_data['nombre'] = nombre
        user_data['apellido'] = apellido
        user_data['rut'] = rut

        with db.get_cursor() as cursor:
            sql = "INSERT INTO users (rut, name, lastname) VALUES (%s, %s, %s)"
            cursor.execute(sql, (rut, nombre, apellido))
            db.database.commit()


        return jsonify({
            "mensaje": "Credenciales almacenadas y guardadas correctamente.",
            "nombre": nombre,
            "apellido": apellido,
            "rut": rut
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/regravedad', methods=['POST'])
def reevaluar_gravedad():
    try:
        data = request.json
        informacion_extra = data.get('informacion_extra', '')

        pregunta_original = user_data.get('pregunta_original', None)
        if not pregunta_original:
            return jsonify({"error": "No se ha recibido una pregunta original previamente."}), 400

        if not informacion_extra:
            return jsonify({"error": "Debes proporcionar 'informacion_extra' para reevaluar la gravedad."}), 400

        pregunta_combinada = f"{pregunta_original}\nInformación adicional: {informacion_extra}"

        query_vector = text_to_vector(pregunta_combinada)
        context_indices = semantic_search(query_vector)
        context = [documents[i] for i in context_indices]

        prompt = f"""
Eres un trabajador de Fltzy, una empresa de gestión de repartidores.
Debes reevaluar la gravedad de un problema considerando tanto la queja inicial como la nueva información adicional proporcionada, además debes solicitar un archivo al usuario, el cual verifique su queja, puede ser tanto un archivo como una imagen.
y sumado a lo anterior dale una respuesta amigable.

Contexto relacionado:
{''.join([f'- {doc}\n' for doc in context])}

Texto combinado del usuario:
{pregunta_combinada}

Entrega solo un valor numérico de gravedad entre 1 y 100:
"""
        response = ask_gemini(prompt)
        match = re.search(r'\b(\d{1,3})\b', response)
        gravedad = match.group(1) if match else "No se pudo evaluar"

        # Mensaje para que el frontend lo muestre al usuario solicitando archivo
        instruccion_usuario = (
            "Por favor, adjunta un archivo o imagen que respalde tu queja para continuar con la gestión."
        )

        return jsonify({
            "pregunta_combinada": pregunta_combinada,
            "gravedad_recalculada": gravedad,
            "contexto_utilizado": context,
            "mensaje_para_usuario": instruccion_usuario
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_data_from_json()
    app.run(debug=True, port=4000)