"""
Example training script for cryptocurrency price prediction models
optimized for Coral TPU
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. This script requires TensorFlow for training.")
    TF_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class CryptoPredictor:
    """
    Cryptocurrency price prediction model designed for Coral TPU deployment
    """
    
    def __init__(self, sequence_length=60, features=16):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = None
        
    def create_model(self, model_type="lstm_classifier"):
        """Create different types of models optimized for Edge TPU"""
        
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for model creation")
        
        if model_type == "lstm_classifier":
            self.model = self._create_lstm_classifier()
        elif model_type == "cnn_classifier":
            self.model = self._create_cnn_classifier()
        elif model_type == "dense_classifier":
            self.model = self._create_dense_classifier()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created {model_type} model with {self.model.count_params()} parameters")
        return self.model
    
    def _create_lstm_classifier(self):
        """Create LSTM-based direction classifier"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.sequence_length, self.features)),
            
            # LSTM layers - keep small for TPU optimization
            keras.layers.LSTM(32, return_sequences=True, dropout=0.2),
            keras.layers.LSTM(16, dropout=0.2),
            
            # Dense layers
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output layer - 3 classes: down, sideways, up
            keras.layers.Dense(3, activation='softmax', name='direction_output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_classifier(self):
        """Create CNN-based direction classifier"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.sequence_length, self.features)),
            
            # Convolutional layers
            keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
            keras.layers.GlobalAveragePooling1D(),
            
            # Dense layers
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output layer
            keras.layers.Dense(3, activation='softmax', name='direction_output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_dense_classifier(self):
        """Create dense-only classifier (fastest for TPU)"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.sequence_length, self.features)),
            keras.layers.Flatten(),
            
            # Dense layers only
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            
            # Output layer
            keras.layers.Dense(3, activation='softmax', name='direction_output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, df, target_column='close'):
        """Prepare data for training"""
        
        # Select features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_12', 'ema_26', 'macd', 'rsi',
            'bb_upper', 'bb_lower', 'bb_position',
            'volume_ratio', 'price_momentum', 'volatility'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if len(available_columns) < 5:
            raise ValueError(f"Insufficient features available: {available_columns}")
        
        logger.info(f"Using features: {available_columns}")
        
        # Get feature data
        feature_data = df[available_columns].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_features, df[target_column].values)
        
        logger.info(f"Created {len(X)} training sequences")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X, y
    
    def _create_sequences(self, features, prices):
        """Create sequences for time series prediction"""
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            # Input sequence
            X.append(features[i-self.sequence_length:i])
            
            # Target: price direction (classification)
            current_price = prices[i-1]
            future_price = prices[i]
            
            if future_price > current_price * 1.01:  # >1% increase
                direction = 2  # up
            elif future_price < current_price * 0.99:  # >1% decrease
                direction = 0  # down
            else:
                direction = 1  # sideways
            
            y.append(direction)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Convert to categorical
        y = keras.utils.to_categorical(y, num_classes=3)
        
        return X, y
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def convert_to_tflite(self, output_path="models/crypto_predictor.tflite", 
                         quantize=True):
        """Convert trained model to TensorFlow Lite for Coral TPU"""
        
        if self.model is None:
            raise ValueError("No trained model to convert")
        
        # Create TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Enable quantization for better TPU performance
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Full integer quantization for Edge TPU
            converter.representative_dataset = self._representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.success(f"Model converted and saved to {output_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return output_path
    
    def _representative_data_gen(self):
        """Generate representative data for quantization"""
        # This should be called with actual training data
        # For now, return dummy data
        for _ in range(100):
            yield [np.random.random((1, self.sequence_length, self.features)).astype(np.float32)]


def load_and_prepare_data(data_path):
    """Load and prepare cryptocurrency data"""
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Add technical indicators if not present
    if 'sma_20' not in df.columns:
        logger.info("Adding technical indicators...")
        df = add_technical_indicators(df)
    
    # Remove rows with missing values
    df = df.dropna()
    
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def add_technical_indicators(df):
    """Add technical indicators to price data"""
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price momentum
    df['price_momentum'] = df['close'].pct_change(periods=5)
    
    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std()
    
    return df


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train crypto prediction model for Coral TPU")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--model-type", choices=["lstm_classifier", "cnn_classifier", "dense_classifier"],
                       default="dense_classifier", help="Type of model to train")
    parser.add_argument("--sequence-length", type=int, default=60, help="Sequence length for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--output", default="models/crypto_predictor.tflite", help="Output TFLite model path")
    
    args = parser.parse_args()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow is required for training. Install with: pip install tensorflow")
        return
    
    try:
        # Load and prepare data
        logger.info("Loading data...")
        df = load_and_prepare_data(args.data)
        
        # Create predictor
        predictor = CryptoPredictor(
            sequence_length=args.sequence_length,
            features=16  # Will be adjusted based on available features
        )
        
        # Prepare training data
        logger.info("Preparing training data...")
        X, y = predictor.prepare_data(df)
        
        # Update features count
        predictor.features = X.shape[2]
        
        # Create model
        logger.info(f"Creating {args.model_type} model...")
        predictor.create_model(args.model_type)
        
        # Train model
        logger.info("Starting training...")
        history = predictor.train(X, y, epochs=args.epochs, batch_size=args.batch_size)
        
        # Evaluate model
        final_accuracy = max(history.history['val_accuracy'])
        logger.success(f"Training completed! Best validation accuracy: {final_accuracy:.4f}")
        
        # Convert to TensorFlow Lite
        logger.info("Converting to TensorFlow Lite...")
        tflite_path = predictor.convert_to_tflite(args.output)
        
        logger.success(f"Model training and conversion completed!")
        logger.info(f"TensorFlow Lite model saved to: {tflite_path}")
        
        # Test the converted model
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            logger.success("✓ TensorFlow Lite model loads successfully")
        except Exception as e:
            logger.error(f"✗ TensorFlow Lite model test failed: {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
