import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, applications
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from PIL import Image
import re
from sklearn.model_selection import train_test_split

# Additional imports for evaluation/plotting.
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

# -----------------------------
# CONFIGURATION & HYPERPARAMETERS
# -----------------------------
VOCAB_SIZE = 5000           # Maximum vocabulary size for the tokenizer.
MAX_CAPTION_LENGTH = 50     # Maximum caption length (in words), capped.
IMAGE_SIZE = (224, 224)     # Input image dimensions for feature extraction.
BATCH_SIZE = 32             # Batch size for the data generator.

# Model hyperparameters:
DROPOUT_RATE = 0.5          # Dropout rate for regularization.
EMBEDDING_DIM = 256         # Dimension of the embedding space.
LSTM_UNITS = 256            # Number of hidden units in the LSTM layer.
LEARNING_RATE = 0.001       # Learning rate for the optimizer.

# -----------------------------
# CAPTION CLEANING & TOKENIZER
# -----------------------------
def load_and_clean_captions(caption_path):
    """
    Loads captions from a CSV file and cleans each caption.
    Assumes the CSV columns are: id, name, caption,
    where the 'name' column holds the image filename (e.g., "12345.jpg").

    The function:
      - Skips the header.
      - Splits each line by commas.
      - Derives the image id from the filename (everything before the '.').
      - Rejoins caption parts (if the caption itself contains commas).
      - Keeps only alphabetic tokens and allowed punctuation symbols ('.', ',', '!', '?').
      - Wraps the caption with 'startseq' at the beginning and 'endseq' at the end.
      - Aggregates all captions by image id in a dictionary.
    """
    captions = {}
    with open(caption_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            image_name = parts[1]
            image_id = image_name.split('.')[0]
            caption = ','.join(parts[2:]).strip().lower()
            caption = ' '.join([word for word in caption.split() 
                                if word.isalpha() or word in ['.', ',', '!', '?']])
            caption = 'startseq ' + caption + ' endseq'
            captions.setdefault(image_id, []).append(caption)
    return captions

def create_tokenizer(captions, vocab_size=VOCAB_SIZE):
    """
    Aggregates all captions and creates a tokenizer.

    Args:
       captions (dict): Mapping from image IDs to lists of captions.
       vocab_size (int): Maximum number of words to keep.
    
    Returns:
       tokenizer: Fitted Keras Tokenizer.
       max_len: Maximum caption length (capped at MAX_CAPTION_LENGTH).
    """
    all_captions = []
    for image_id in captions:
        all_captions.extend(captions[image_id])
    
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<unk>",
        filters=''  # No additional filtering (captions are pre-cleaned)
    )
    tokenizer.fit_on_texts(all_captions)
    
    max_len = max(len(caption.split()) for caption in all_captions)
    max_len = min(max_len, MAX_CAPTION_LENGTH)
    
    return tokenizer, max_len

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
class FeatureExtractor:
    def _init_(self, weights_path=None):
        self.model = None
        self.weights_path = weights_path
        
    def get_model(self):
        if self.model is None:
            base_model = EfficientNetV2B0(
                weights=self.weights_path if self.weights_path else 'imagenet',
                include_top=False,
                pooling='avg'
            )
            self.model = models.Model(
                inputs=base_model.input,
                outputs=base_model.output
            )
            self.model.trainable = False
        return self.model
    
    def extract_features(self, image_path, image_id, augment=False):
        try:
            img_path = os.path.join(image_path, image_id + '.jpg')
            img = Image.open(img_path)
            if augment:
                if random.random() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.8:
                    img = img.rotate(random.uniform(-10, 10))
            img = img.resize(IMAGE_SIZE)
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img = img.reshape((1, *img.shape))
            img = applications.efficientnet_v2.preprocess_input(img)
            features = self.get_model().predict(img, verbose=0)
            return features[0]
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            return None

feature_extractor = FeatureExtractor()

# -----------------------------
# ATTENTION LAYERS
# -----------------------------
class ChannelAttention(layers.Layer):
    def _init_(self, ratio=8):
        super(ChannelAttention, self)._init_()
        self.ratio = ratio
        self.gap = layers.GlobalAveragePooling1D()
        self.gmp = layers.GlobalMaxPooling1D()
        self.shared_mlp = tf.keras.Sequential([
            layers.Dense(units=1280 // self.ratio, activation='relu'),
            layers.Dense(units=1280)
        ])
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, channels)
        gap = self.gap(inputs)
        gmp = self.gmp(inputs)
        gap_mlp = self.shared_mlp(gap)
        gmp_mlp = self.shared_mlp(gmp)
        channel_attention = self.sigmoid(gap_mlp + gmp_mlp)
        return inputs * channel_attention[:, None, :]

class SpatialAttention(layers.Layer):
    def _init_(self):
        super(SpatialAttention, self)._init_()
        self.conv = layers.Conv1D(1, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, inputs):
        spatial_attention = self.conv(inputs)
        return inputs * spatial_attention

# -----------------------------
# DATA GENERATOR
# -----------------------------
class CaptioningDataGenerator(Sequence):
    def _init_(self, image_ids, captions, tokenizer, max_length, vocab_size, 
                 batch_size, image_path, augment=False):
        self.image_ids = image_ids
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.image_path = image_path
        self.augment = augment
        self.feature_cache = {} 
        
    def _len_(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))
    
    def _getitem_(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        X1, X2, y = [], [], []
        for image_id in batch_ids:
            if image_id not in self.feature_cache:
                features = feature_extractor.extract_features(self.image_path, image_id, augment=self.augment)
                if features is not None:
                    self.feature_cache[image_id] = features
                else:
                    continue
            features = self.feature_cache[image_id]
            for caption in self.captions.get(image_id, []):
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(features)
                    X2.append(in_seq)
                    y.append(out_seq)
                    
        if len(X1) == 0:
            return self._getitem((idx + 1) % self.len_())
        
        indices = np.arange(len(X1))
        np.random.shuffle(indices)
        X1 = np.array(X1)[indices]
        X2 = np.array(X2)[indices]
        y = np.array(y)[indices]
        return (X1, X2), y

# -----------------------------
# IMPROVED IMAGE-CAPTIONING MODEL WITH DUAL ATTENTION
# -----------------------------
def create_improved_model(vocab_size, max_length):
    # Image feature branch with dual attention.
    input1 = layers.Input(shape=(1280,))
    reshaped = layers.Reshape((1, 1280))(input1)
    channel_attention = ChannelAttention()(reshaped)
    spatial_attention = SpatialAttention()(channel_attention)
    attended_features = layers.Flatten()(spatial_attention)
    fe1 = layers.Dropout(DROPOUT_RATE)(attended_features)
    fe2 = layers.Dense(512, activation='relu')(fe1)
    fe3 = layers.Dropout(DROPOUT_RATE)(fe2)
    fe4 = layers.Dense(256, activation='relu')(fe3)
    
    # Text branch.
    input2 = layers.Input(shape=(max_length,))
    se1 = layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=False)(input2)
    se2 = layers.Dropout(DROPOUT_RATE)(se1)
    se3 = layers.Bidirectional(layers.LSTM(LSTM_UNITS))(se2)
    
    # Merge image and text features.
    decoder1 = layers.Concatenate()([fe4, se3])
    decoder2 = layers.Dense(512, activation='relu')(decoder1)
    decoder3 = layers.Dropout(DROPOUT_RATE)(decoder2)
    decoder4 = layers.Dense(256, activation='relu')(decoder3)
    outputs = layers.Dense(vocab_size, activation='softmax')(decoder4)
    
    model = models.Model(inputs=[input1, input2], outputs=outputs)
    optimizer = optimizers.Adam(
        learning_rate=LEARNING_RATE,
        decay=1e-6
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

# -----------------------------
# CAPTION GENERATION & EVALUATION FUNCTIONS
# -----------------------------
smoothie = SmoothingFunction().method1
NUM_SAMPLES = 4   # Number of sample images to display

def generate_caption(model, tokenizer, image_features, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.array([image_features]), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq ', '')

def display_results(model, tokenizer, test_ids, captions, max_length, n=NUM_SAMPLES):
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(n*2, 1, height_ratios=[3, 1]*n, hspace=0.4)
    displayed = 0
    attempts = 0
    max_attempts = min(200, len(test_ids))
    global test_img_dir  # Using test image directory
    while displayed < n and attempts < max_attempts:
        attempts += 1
        image_id = random.choice(test_ids)
        features = feature_extractor.extract_features(test_img_dir, image_id)
        if features is None:
            continue
        gen_caption = generate_caption(model, tokenizer, features, max_length)
        ref_captions = [c.replace('startseq ', '').replace(' endseq', '') for c in captions[image_id]]
        gen_tokens = gen_caption.split()
        ref_tokens = [ref.split() for ref in ref_captions]
        bleu_score = sentence_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        ax_img = fig.add_subplot(gs[displayed*2])
        img = Image.open(os.path.join(test_img_dir, image_id + '.jpg'))
        img_aspect = img.size[0] / img.size[1]
        if img_aspect > 1.5:
            fig.set_size_inches(12, fig.get_size_inches()[1])
            ax_img.imshow(img)
        else:
            ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f"Image {displayed+1} (BLEU: {bleu_score:.4f})", pad=10)
        ax_cap = fig.add_subplot(gs[displayed*2 + 1])
        ax_cap.axis('off')
        caption_text = f"Generated: {gen_caption}"
        ax_cap.text(0.5, 0.5, caption_text, ha='center', va='center', fontsize=24,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        if displayed < n - 1:
            fig.subplots_adjust(hspace=0.3)
        displayed += 1
    if displayed == 0:
        print("No images found")
    else:
        plt.tight_layout()
        plt.show()

def evaluate_model(model, tokenizer, test_ids, captions, max_length, sample_size=500):
    actual, predicted = [], []
    test_subset = test_ids[:sample_size]
    for image_id in tqdm(test_subset, desc="Evaluating"):
        features = feature_extractor.extract_features(test_img_dir, image_id)
        if features is None:
            continue
        yhat = generate_caption(model, tokenizer, features, max_length)
        references = [c.replace('startseq ', '').replace(' endseq', '') for c in captions[image_id]]
        actual.append(references)
        predicted.append(yhat)
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    print("\nModel Evaluation Results:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    return bleu1, bleu2, bleu3, bleu4

# -----------------------------
# HELPER: FILTER AVAILABLE IMAGES
# -----------------------------
def filter_available_images(captions_dict, image_dir):
    """
    Check if the image file exists for each key in the captions dictionary.
    Returns a new dictionary filtered to only include available images.
    """
    filtered = {}
    missing_count=0
    for img_id, caps in captions_dict.items():
        file_path = os.path.join(image_dir, img_id + '.jpg')
        if os.path.exists(file_path):
            filtered[img_id] = caps
        else:
            # print(f"Missing file: {file_path}")
            missing_count+=1
    return filtered

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if _name_ == '_main_':
    # Define CSV annotation paths.
    train_csv = "/kaggle/input/roco-dataset/all_data/train/radiology/traindata.csv"
    val_csv   = "/kaggle/input/roco-dataset/all_data/validation/radiology/valdata.csv"
    test_csv  = "/kaggle/input/roco-dataset/all_data/test/radiology/testdata.csv"
    
    # Define image directories.
    train_img_dir = "/kaggle/input/roco-dataset/all_data/train/radiology/images"
    val_img_dir   = "/kaggle/input/roco-dataset/all_data/validation/radiology/images"
    test_img_dir  = "/kaggle/input/roco-dataset/all_data/test/radiology/images"
    
    # Load and clean captions.
    captions_train = load_and_clean_captions(train_csv)
    captions_val   = load_and_clean_captions(val_csv)
    captions_test  = load_and_clean_captions(test_csv)
    
    # Filter out entries for which the image file is missing.
    captions_train = filter_available_images(captions_train, train_img_dir)
    captions_val   = filter_available_images(captions_val, val_img_dir)
    captions_test  = filter_available_images(captions_test, test_img_dir)
    
    # Get image IDs for each split.
    image_ids_train = list(captions_train.keys())
    image_ids_val   = list(captions_val.keys())
    image_ids_test  = list(captions_test.keys())
    
    print("Available training images:", len(image_ids_train))
    print("Available validation images:", len(image_ids_val))
    print("Available testing images:", len(image_ids_test))
    
    # Create tokenizer and determine maximum caption length.
    tokenizer, max_length = create_tokenizer(captions_train)
    vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_SIZE)
    print("Vocabulary Size:", vocab_size, "Max Caption Length:", max_length)
    
    # Instantiate data generators.
    train_generator = CaptioningDataGenerator(
        image_ids_train, captions_train,
        tokenizer, max_length, vocab_size, BATCH_SIZE,
        train_img_dir, augment=True
    )
    val_generator = CaptioningDataGenerator(
        image_ids_val, captions_val,
        tokenizer, max_length, vocab_size, BATCH_SIZE,
        val_img_dir, augment=False
    )
    test_generator = CaptioningDataGenerator(
        image_ids_test, captions_test,
        tokenizer, max_length, vocab_size, BATCH_SIZE,
        test_img_dir, augment=False
    )
    
    # Create the improved model.
    model = create_improved_model(vocab_size, max_length)
    model.summary()
    
    # -----------------------------
    # CALLBACKS & TRAINING
    # -----------------------------
    callbacks = [
        ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=test_generator,
        callbacks=callbacks,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Plot training vs. validation loss.
    plt.figure(figsize=(15, 7), dpi=200)
    sns.set_style('whitegrid')
    plt.plot([x+1 for x in range(len(history.history['loss']))],
             history.history['loss'], color='#004EFF', marker='o')
    plt.plot([x+1 for x in range(len(history.history['loss']))],
             history.history['val_loss'], color='#00008B', marker='h')
    plt.title('Train VS Validation', fontsize=15, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend(['Train Loss', 'Validation Loss'], loc='best')
    plt.show()
    
    # -----------------------------
    # INFERENCE & EVALUATION
    # -----------------------------
    print("Displaying sample results...")
    display_results(model, tokenizer, image_ids_test, captions_test, max_length)
    
    print("\nEvaluating model on test set...")
    bleu_scores = evaluate_model(model, tokenizer, image_ids_test, captions_test, max_length)
    print("\nTraining and evaluation complete!")