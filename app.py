import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# -------------------------------------------------------------
# CONFIG STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="HealthHelp IA",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0f0f0f; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# CRIAR MODELO DE EMOÇÕES (FALLBACK/ESTRUTURA)
# -------------------------------------------------------------
def criar_modelo_emocoes():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -------------------------------------------------------------
# CARREGAR MODELO CORRETO
# -------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    # Tenta carregar o arquivo que você já tem treinado
    arquivo_modelo = "emotion_model.h5"
    
    if os.path.exists(arquivo_modelo):
        try:
            model = tf.keras.models.load_model(arquivo_modelo)
            return model
        except Exception as e:
            st.error(f"Erro ao ler o arquivo do modelo: {e}")
            return None
    else:
        # Se não achar o arquivo treinado, avisa e cria um vazio (apenas para não quebrar o app)
        st.warning("⚠️ AVISO: O arquivo 'emotion_model.h5' não foi encontrado. Usando modelo não treinado (resultados serão aleatórios). Verifique se o arquivo está na pasta.")
        model = criar_modelo_emocoes()
        return model

emotion_model = carregar_modelo()
emotion_labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]


# -------------------------------------------------------------
# FUNÇÃO DE DETECÇÃO DE EMOÇÃO COM RECORTE DE ROSTO
# -------------------------------------------------------------
def detectar_emocao(image):
    # Carrega classificador de rosto (padrão do OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Se vier RGBA (png) → converter para RGB
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # RGB → BGR (padrão OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Escala de Cinza
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # DETECÇÃO DE ROSTO
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Se não achar rosto, retorna aviso
    if len(faces) == 0:
        return None, None

    # Pega o maior rosto encontrado (caso tenha mais de um, foca no primeiro)
    x, y, w, h = faces[0]
    
    # Desenha um retângulo na imagem original para mostrar onde achou o rosto (opcional, mas legal visualmente)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Recorta a região do rosto (ROI)
    roi_gray = img_gray[y:y+h, x:x+w]

    # Redimensiona para 48x48 (tamanho que a IA espera)
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalizar (0 a 1)
    roi_gray = roi_gray.astype("float32") / 255.0

    # Ajustar formato para o Keras: (1, 48, 48, 1)
    roi_gray = np.expand_dims(roi_gray, axis=-1)
    roi_gray = np.expand_dims(roi_gray, axis=0)

    # Predição
    preds = emotion_model.predict(roi_gray)[0]
    emotion = emotion_labels[np.argmax(preds)]

    return emotion, preds


# -------------------------------------------------------------
# SUGESTÕES BASEADAS NA EMOÇÃO
# -------------------------------------------------------------
def sugestoes_emocao(emocao):
    base = {
        "Feliz": [
            "Mantenha os bons hábitos!",
            "Aproveite para iniciar um novo hábito positivo.",
            "Compartilhe algo positivo com alguém."
        ],
        "Triste": [
            "Separe 30 minutos para algo que te faça bem.",
            "Faça pausas durante o dia.",
            "Se permanecer triste, considere falar com alguém de confiança."
        ],
        "Raiva": [
            "Tente exercícios de respiração.",
            "Evite ambientes muito estressantes.",
            "Uma caminhada leve pode ajudar."
        ],
        "Medo": [
            "Liste suas preocupações.",
            "Evite telas antes de dormir.",
            "Faça respiração profunda por 2 minutos."
        ],
        "Surpreso": [
            "Reorganize sua agenda.",
            "Tente manter horários mais fixos.",
            "Faça uma pausa rápida."
        ],
        "Nojo": [
            "Divida tarefas desagradáveis ao longo do dia.",
            "Recompense-se após tarefas difíceis.",
            "Tire pequenas pausas."
        ],
        "Neutro": [
            "Inclua algo divertido no seu dia.",
            "Defina uma mini-meta simples.",
            "Hidrate-se e alongue-se."
        ]
    }
    return base.get(emocao, ["Cuide-se e mantenha equilíbrio."])


# -------------------------------------------------------------
# ANÁLISE DE ROTINA
# -------------------------------------------------------------
def analisar_rotina(sono, trabalho, lazer, exercicio):
    feedback = []

    if sono < 7:
        feedback.append("Você dormiu pouco. O ideal é 7 a 8 horas.")
    elif sono > 9:
        feedback.append("Sono acima da média. Pode ser cansaço acumulado.")
    else:
        feedback.append("Seu sono está equilibrado!")

    if trabalho > 9:
        feedback.append("Carga alta de trabalho. Faça pausas estratégicas.")
    else:
        feedback.append("Boa quantidade de trabalho/estudo.")

    if lazer < 1:
        feedback.append("Pouco lazer. Inclua atividades que te fazem bem.")
    else:
        feedback.append("Ótimo! Você reservou tempo para lazer.")

    if exercicio == 0:
        feedback.append("Tente ao menos 10 minutos de caminhada hoje.")
    else:
        feedback.append("Boa! Atividade física faz bem ao humor.")

    return feedback


# -------------------------------------------------------------
# INTERFACE
# -------------------------------------------------------------
st.title("🧠 HealthHelp IA")
st.write("Aplicativo de análise emocional e hábitos usando Deep Learning.")

tabs = st.tabs(["📸 Análise de Emoções", "📆 Avaliação de Rotina"])


# =============================================================
# ABA 1 — ANÁLISE DE EMOÇÕES
# =============================================================
with tabs[0]:

    st.subheader("Envie uma foto do seu rosto")
    st.info("Dica: Tente usar uma foto bem iluminada e onde seu rosto esteja visível.")

    foto = st.file_uploader("Formatos aceitos: PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])

    if foto:
        # Carrega a imagem
        image_pil = Image.open(foto)
        img_np = np.array(image_pil)
        
        # Exibe a imagem original
        st.image(image_pil, caption="Imagem enviada", use_container_width=True)

        if st.button("Analisar emoções"):
            with st.spinner("Detectando rosto e analisando emoção..."):
                
                emocao, probs = detectar_emocao(img_np)

                if emocao is None:
                    st.error("⚠️ Não foi possível detectar um rosto na imagem. Tente outra foto mais clara ou mais próxima.")
                else:
                    st.success(f"🎭 Emoção predominante: **{emocao}**")

                    # Colunas para exibir gráfico e dicas
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Probabilidades:")
                        # Ordena para mostrar as maiores probabilidades primeiro
                        sorted_indices = np.argsort(probs)[::-1]
                        for i in sorted_indices:
                            label = emotion_labels[i]
                            p = probs[i]
                            if p > 0.01: # Só mostra se tiver mais de 1% de chance
                                st.progress(float(p))
                                st.write(f"{label}: {p*100:.1f}%")

                    with col2:
                        st.markdown("### 💡 Recomendações:")
                        for dica in sugestoes_emocao(emocao):
                            st.info(f"- {dica}")


# =============================================================
# ABA 2 — ROTINA
# =============================================================
with tabs[1]:

    st.subheader("Como está sua rotina hoje?")

    sono = st.slider("Horas de sono", 0, 12, 7)
    trabalho = st.slider("Horas de trabalho/estudo", 0, 14, 8)
    lazer = st.slider("Horas de lazer", 0, 8, 1)
    exercicio = st.slider("Horas de exercício", 0, 4, 0)

    if st.button("Analisar rotina"):
        st.markdown("### 📋 Resultados:")

        feedback = analisar_rotina(sono, trabalho, lazer, exercicio)

        for f in feedback:
            st.markdown(f"- {f}")

        st.markdown("### ✨ Dica Final:")
        st.write("Tente registrar sua rotina diariamente para acompanhar sua evolução.")
