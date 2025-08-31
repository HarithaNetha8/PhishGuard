import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import logging
from PIL import Image, ImageDraw, ImageFont
import random

# Configure logging
logging.basicConfig(level=logging.INFO)

def generate_sample_images(n_samples=100, image_size=(64, 64)):
    """
    Generate sample images for demonstration
    Creates synthetic website-like images with different characteristics for phishing vs legitimate
    """
    images = []
    labels = []
    
    # Create directories for sample images
    os.makedirs('sample_images/phishing', exist_ok=True)
    os.makedirs('sample_images/legitimate', exist_ok=True)
    
    for i in range(n_samples):
        # Generate label (0 = legitimate, 1 = phishing)
        is_phishing = random.choice([0, 1])
        
        # Create image
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        if is_phishing:
            # Phishing characteristics
            # Use more aggressive colors, suspicious text patterns
            bg_color = random.choice(['red', 'orange', 'yellow', 'magenta'])
            
            # Draw suspicious elements
            draw.rectangle([10, 10, image_size[0]-10, 50], fill=bg_color)
            
            # Add suspicious text patterns
            try:
                # Simple text without font
                draw.text((20, 20), "URGENT! Click here!", fill='black')
                draw.text((20, 60), "Limited time offer!", fill='red')
                draw.text((20, 100), "Enter your password:", fill='black')
            except:
                pass
            
            # Add more visual noise
            for _ in range(random.randint(5, 15)):
                x = random.randint(0, image_size[0])
                y = random.randint(0, image_size[1])
                w = random.randint(10, 50)
                h = random.randint(10, 50)
                color = random.choice(['red', 'yellow', 'orange', 'cyan'])
                draw.rectangle([x, y, x+w, y+h], fill=color)
            
            folder = 'phishing'
        else:
            # Legitimate characteristics
            # Use professional colors, clean layout
            bg_color = random.choice(['lightblue', 'lightgray', 'white', 'lightgreen'])
            
            # Draw professional elements
            draw.rectangle([10, 10, image_size[0]-10, 50], fill=bg_color)
            
            # Add professional text
            try:
                draw.text((20, 20), "Welcome to our service", fill='black')
                draw.text((20, 60), "Professional website", fill='blue')
                draw.text((20, 100), "About us | Contact", fill='gray')
            except:
                pass
            
            # Add minimal visual elements
            for _ in range(random.randint(1, 5)):
                x = random.randint(0, image_size[0])
                y = random.randint(0, image_size[1])
                w = random.randint(20, 80)
                h = random.randint(20, 80)
                color = random.choice(['lightblue', 'lightgray', 'white'])
                draw.rectangle([x, y, x+w, y+h], fill=color)
            
            folder = 'legitimate'
        
        # Save image
        img_path = f'sample_images/{folder}/sample_{i}_{is_phishing}.png'
        img.save(img_path)
        
        # Convert to array
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)
        labels.append(is_phishing)
    
    return np.array(images), np.array(labels)

def load_or_generate_images():
    """
    Load existing images or generate sample data
    """
    # Check if sample images exist
    if (os.path.exists('sample_images/phishing') and 
        os.path.exists('sample_images/legitimate') and
        len(os.listdir('sample_images/phishing')) > 0 and
        len(os.listdir('sample_images/legitimate')) > 0):
        
        logging.info("Loading existing sample images...")
        
        # Load images from directories
        images = []
        labels = []
        
        # Load phishing images
        phishing_dir = 'sample_images/phishing'
        for filename in os.listdir(phishing_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(phishing_dir, filename)
                try:
                    img = Image.open(img_path)
                    img = img.resize((224, 224))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(img_array)
                    labels.append(1)  # phishing
                except Exception as e:
                    logging.warning(f"Error loading {img_path}: {e}")
        
        # Load legitimate images
        legitimate_dir = 'sample_images/legitimate'
        for filename in os.listdir(legitimate_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(legitimate_dir, filename)
                try:
                    img = Image.open(img_path)
                    img = img.resize((224, 224))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(img_array)
                    labels.append(0)  # legitimate
                except Exception as e:
                    logging.warning(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    else:
        logging.info("Generating sample images...")
        return generate_sample_images()

def create_cnn_model(input_shape=(64, 64, 3)):
    """
    Create a CNN model for screenshot-based phishing detection
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Global average pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_cnn_model():
    """
    Train CNN model for screenshot-based phishing detection
    """
    # Load or generate images
    X, y = load_or_generate_images()
    
    logging.info(f"Dataset shape: {X.shape}")
    logging.info(f"Phishing samples: {np.sum(y)}, Legitimate samples: {len(y) - np.sum(y)}")
    
    if len(X) == 0:
        raise Exception("No images found or generated")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    logging.info("Model architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train model
    logging.info("Training CNN model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/cnn_screenshot_model.h5'
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    return model, test_accuracy, history

def main():
    """Main function to train the screenshot-based phishing detection model"""
    try:
        model, accuracy, history = train_cnn_model()
        print(f"Screenshot model training completed successfully! Accuracy: {accuracy:.4f}")
        return True
    except Exception as e:
        logging.error(f"Error training screenshot model: {str(e)}")
        return False

if __name__ == "__main__":
    main()
