from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

model = MobileNetV2(weights='imagenet')  # Pretrained model

# Dummy labels to match your expected format
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

print("model loaded.")

