import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

#włączenie pełnej szerokosci strony
st.set_page_config(layout="wide")

#komunikat
st.subheader(":blue[Liczenie i detekcja] osób na zdjęciach grupowych :female-detective:")
st.write("""Wybierz dowolne zdjęcie grupowe i pozwól, aby program policzył za Ciebie, ile osób się na nim znajduje.
Strona powstała, jako część projektu z CPO, a jej autorem jest Alicja :star2:""")

#wczytanie zdjęc
plik = st.file_uploader("Wczytaj zdjęcie", type=["jpg", "jpeg", "png"])
if plik is not None:
    st.success("Zdjęcie zostało załadowane poprawnie!")

#konwersja pliku do formatu obrazu
    obraz = Image.open(plik)
    obraz = np.array(obraz)
    obraz = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)

    img_g = cv2.cvtColor(obraz, cv2.COLOR_RGB2GRAY)

#checkboxy do wyboru metody detekcji
    st.write(""":blue[Zdecyduj, czy chcesz przetworzyć zdjęcie przed detekcją osób.
    ***Zaznacz tylko jedną opcję!***]""")
    equalize = st.checkbox("Wyrównać histogram przed detekcją?")
    no_equalize = st.checkbox("Wykonać detekcję bez wyrównania histogramu?")

#wybor przetworzenia zdjecia
    metoda_histogramu = "bez wyrównania histogramu"

    if equalize and no_equalize:
        st.warning("Zaznacz tylko jedną opcję: z wyrównaniem histogramu **lub** bez wyrównania.")
        st.stop()
    elif equalize:
        img_g = cv2.equalizeHist(img_g)
        metoda_histogramu = "po wyrównaniu histogramu"

    if not equalize:
        st.warning("Nie wybrano żadnej opcji!")

#zdefiniowanie metody detekcji
    metoda_detekcji = st.selectbox("Wybierz metodę detekcji:",
    ("Haar_faces", "Haar_eyes", "Haar_bodies", "HOG"),
    index=None,
    placeholder="Metoda...")


    wykryte_osoby = []
    if metoda_detekcji in ["Haar_faces", "Haar_eyes", "Haar_bodies"]:
        minNs = st.number_input(
        """Określ parametr minNeighbors.
Jeśli chcesz dowiedzieć się więcej, przejdź na stronę [dokumentacji OpenCV](https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html).
\n***Zalecane wartości: 3 lub 5***""",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        format="%d")
        st.write("Pamiętaj, aby wcisnąć :red[***ENTER***] po zmianie wartości minNeighbors!")

        folder_haar = os.path.join(os.path.dirname(__file__), 'haar')


        if metoda_detekcji == "Haar_faces":
            cascade_path = os.path.join(folder_haar, "haarcascade_frontalface_alt2.xml")
        elif metoda_detekcji == "Haar_eyes":
            cascade_path = os.path.join(folder_haar, "haarcascade_eye.xml")
        else:
            cascade_path = os.path.join(folder_haar, "haarcascade_fullbody.xml")

        klasyfikator = cv2.CascadeClassifier(cascade_path)
        wykryte_osoby = klasyfikator.detectMultiScale(img_g, scaleFactor=1.1, minNeighbors=int(minNs))

    elif metoda_detekcji == "HOG":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        wykryte_osoby, _ = hog.detectMultiScale(img_g, winStride=(8, 8), padding=(8, 8), scale=1.05)


    img_d = obraz.copy()
    for (x, y, w, h) in wykryte_osoby:
        cv2.rectangle(img_d, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_s = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    col1.image(img_g, caption=f"Obraz użyty do detekcji {metoda_histogramu}")
    col2.image(img_s, caption=f"Liczba wykrytych osób za pomocą algorytmu {metoda_detekcji}: {len(wykryte_osoby)}")

#ocena strony
st.write("---")
st.write(":blue[Podziel się swoją opinią o tej stronie!]")
sentiment_mapping = ["1", "2", "3", "4", "5"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"Twoja ocena to: {sentiment_mapping[selected]} :star2:")



