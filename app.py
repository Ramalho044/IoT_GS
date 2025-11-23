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
# CRIAR MODELO DE EMOÇÕES (FALLBACK)
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# -------------------------------------------------------------
# CARREGAR MODELO (CORREÇÃO DE COMPATIBILIDADE)
# -------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    arquivo_modelo = "emotion_model.h5"
    
    if os.path.exists(arquivo_modelo):
        try:
            # compile=False é CRÍTICO para evitar erros de versão entre onde foi treinado e o Streamlit
            model = tf.keras.models.load_model(arquivo_modelo, compile=False)
            
            # Recompilamos manualmente apenas para garantir
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
            return None
    else:
        st.warning("⚠️ 'emotion_model.h5' não encontrado. Usando modelo vazio (aleatório).")
        model = criar_modelo_emocoes()
        return model

emotion_model = carregar_modelo()
emotion_labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]


# -------------------------------------------------------------
# FUNÇÃO DE DETECÇÃO (CORREÇÃO DE SHAPE DINÂMICO)
# -------------------------------------------------------------
def detectar_emocao(image):
    # Carrega classificador de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Converter para RGB se necessário
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # BGR e Gray para detecção
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detecta rostos
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None

    # Pega o primeiro rosto
    x, y, w, h = faces[0]
    
    # Desenha retângulo visual
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Recorta ROI (rosto)
    roi_gray = img_gray[y:y+h, x:x+w]

    # --- TRATAMENTO DINÂMICO DE ENTRADA ---
    try:
        # Tenta pegar o formato que o modelo espera (ex: 48x48x1 ou 48x48x3)
        input_shape = emotion_model.input_shape
        # input_shape geralmente é (None, Altura, Largura, Canais)
        req_height = input_shape[1]
        req_width = input_shape[2]
        req_channels = input_shape[3]
    except:
        # Padrão caso não consiga ler
        req_height, req_width = 48, 48
        req_channels = 1

    # 1. Redimensionar para o tamanho que o modelo quer
    roi_resized = cv2.resize(roi_gray, (req_width, req_height), interpolation=cv2.INTER_AREA)
    
    # 2. Normalizar
    roi_float = roi_resized.astype("float32") / 255.0

    # 3. Ajustar Canais (1 para Grayscale, 3 para RGB)
    if req_channels == 1:
        # Adiciona dimensão de canal (48, 48) -> (48, 48, 1)
        roi_final = np.expand_dims(roi_float, axis=-1)
    else:
        # Se o modelo quiser RGB, convertemos o Grayscale de volta para RGB
        roi_final = cv2.cvtColor(roi_float, cv2.COLOR_GRAY2RGB)

    # 4. Adicionar Batch Dimension: (1, 48, 48, 1)
    roi_final = np.expand_dims(roi_final, axis=0)

    # Predição
    preds = emotion_model.predict(roi_final)[0]
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
        
        # Exibe a imagem original com o retângulo se detectado
        col_img, col_res = st.columns([1, 2])
        
        if st.button("Analisar emoções"):
            with st.spinner("Detectando rosto e analisando emoção..."):
                
                emocao, probs = detectar_emocao(img_np)

                if emocao is None:
                    st.error("⚠️ Não foi possível detectar um rosto na imagem. Tente uma foto mais próxima e frontal.")
                    st.image(image_pil, caption="Imagem Original", width=300)
                else:
                    # Exibe a imagem com o rosto marcado (img_np foi modificado pela função detectar_emocao com o retângulo)
                    st.image(img_np, caption="Rosto Detectado", width=300)
                    
                    st.success(f"🎭 Emoção predominante: **{emocao}**")

                    st.markdown("### 📊 Probabilidades:")
                    # Ordena para mostrar as maiores probabilidades primeiro
                    sorted_indices = np.argsort(probs)[::-1]
                    for i in sorted_indices:
                        label = emotion_labels[i]
                        p = probs[i]
                        if p > 0.01: # Só mostra se tiver mais de 1%
                            st.progress(float(p))
                            st.write(f"{label}: {p*100:.1f}%")

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
