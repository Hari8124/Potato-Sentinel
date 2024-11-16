import tensorflow as tf

# Load the ensemble model
try:
    print("Loading ensemble model...")
    ensemble_model = tf.keras.models.load_model('ensemble_model.h5')
    print("Model loaded successfully.\n")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Print the model summary
print("Model Summary:\n")
ensemble_model.summary()

# Print the layers of the model
print("\nModel Layers:\n")
for i, layer in enumerate(ensemble_model.layers):
    print(f"Layer {i+1}: {layer.name} - {layer.__class__.__name__}")
