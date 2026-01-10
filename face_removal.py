import cv2
import numpy as np
from tqdm import tqdm
import os

# Tentar importar MediaPipe (versão mais recente)
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = hasattr(mp, 'solutions')
    if MEDIAPIPE_AVAILABLE:
        print(" MediaPipe disponível - usando MediaPipe Face Detection")
    else:
        print(" MediaPipe sem 'solutions' - usando OpenCV Haar Cascade")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print(" MediaPipe não disponível - usando OpenCV Haar Cascade")


class FaceRemover:
    """Remove rostos de imagens para focar na postura corporal"""

    def __init__(self, blur_intensity=50, use_mediapipe=True):
        """
        Inicializa o removedor de rostos

        Args:
            blur_intensity: intensidade do blur (deve ser ímpar)
            use_mediapipe: tentar usar MediaPipe se disponível
        """
        self.blur_intensity = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE

        if self.use_mediapipe:
            # Usar MediaPipe
            self.mp_face = mp.solutions.face_detection
            self.face_detection = self.mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            print(" Usando MediaPipe para deteção facial")
        else:
            # Usar OpenCV Haar Cascade como fallback
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                raise Exception("Erro ao carregar Haar Cascade para deteção facial")

            print(" Usando OpenCV Haar Cascade para deteção facial")

    def remove_face_mediapipe(self, image):
        """Remove rosto usando MediaPipe"""
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                # Garantir coordenadas dentro da imagem
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + box_w)
                y2 = min(h, y + box_h)

                # Aplicar blur
                face_region = image[y:y2, x:x2]
                if face_region.size > 0:
                    blurred = cv2.GaussianBlur(
                        face_region,
                        (self.blur_intensity, self.blur_intensity),
                        0
                    )
                    image[y:y2, x:x2] = blurred

        return image

    def remove_face_opencv(self, image):
        """Remove rosto usando OpenCV Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detetar rostos
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Aplicar blur em cada rosto detetado
        for (x, y, w, h) in faces:
            # Expandir um pouco a região (10% em cada direção)
            margin = int(0.1 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + 2 * margin
            h = h + 2 * margin

            # Garantir que não ultrapassa limites da imagem
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)

            face_region = image[y:y2, x:x2]

            if face_region.size > 0:
                blurred = cv2.GaussianBlur(
                    face_region,
                    (self.blur_intensity, self.blur_intensity),
                    0
                )
                image[y:y2, x:x2] = blurred

        return image

    def remove_face(self, image):
        """
        Remove o rosto da imagem através de blur

        Args:
            image: numpy array (BGR format)

        Returns:
            image com rosto removido
        """
        if self.use_mediapipe:
            return self.remove_face_mediapipe(image)
        else:
            return self.remove_face_opencv(image)

    def process_batch(self, image_paths, output_dir):
        """
        Processa um lote de imagens

        Args:
            image_paths: lista de caminhos para imagens
            output_dir: diretório de saída
        """
        os.makedirs(output_dir, exist_ok=True)

        processed_count = 0
        failed_count = 0

        for img_path in tqdm(image_paths, desc="Removendo rostos"):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    processed = self.remove_face(img)

                    # Salvar com mesmo nome
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, processed)
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"\nErro ao processar {img_path}: {e}")
                failed_count += 1

        print(f"\n Processadas: {processed_count}")
        if failed_count > 0:
            print(f" Falhadas: {failed_count}")

    def __del__(self):
        """Limpar recursos"""
        if self.use_mediapipe and hasattr(self, 'face_detection'):
            try:
                self.face_detection.close()
            except:
                pass


# Teste rápido
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" TESTE DO MÓDULO DE REMOÇÃO DE ROSTOS")
    print("=" * 70 + "\n")

    try:
        # Tentar criar o removedor
        remover = FaceRemover()
        print(" FaceRemover inicializado com sucesso")

        # Criar uma imagem de teste
        print("\nCriando imagem de teste...", end=" ")
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("")

        # Tentar processar
        print("Testando remoção de rostos...", end=" ")
        result = remover.remove_face(test_img.copy())
        print("")

        print("\n TESTE PASSOU! Módulo funcionando corretamente.")

        # Limpar
        del remover

    except Exception as e:
        print(f"\n ERRO NO TESTE: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)