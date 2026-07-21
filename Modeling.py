import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU
from preprocessing import load_and_preprocess_data


def build_lstm_model(input_shape: tuple, num_classes: int) -> Sequential:
    return Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=num_classes, activation='softmax')
    ], name="LSTM_Model")


def build_ann_model(input_shape: tuple, num_classes: int) -> Sequential:
    return Sequential([
        Input(shape=input_shape),
        Dense(units=128, activation='relu'),
        Dropout(0.3),
        Dense(units=64, activation='relu'),
        Dropout(0.3),
        Dense(units=num_classes, activation='softmax')
    ], name="ANN_Model")


def build_gru_model(input_shape: tuple, num_classes: int) -> Sequential:
    return Sequential([
        Input(shape=input_shape),
        GRU(units=50, return_sequences=True),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=num_classes, activation='softmax')
    ], name="GRU_Model")


def build_hybrid_model(input_shape: tuple, num_classes: int) -> Sequential:
    return Sequential([
        Input(shape=input_shape),
        GRU(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(units=num_classes, activation='softmax')
    ], name="Hybrid_GRU_LSTM_Model")


def train_and_evaluate():
    # Load dataset from preprocessing module
    data = load_and_preprocess_data()
    
    X_train_ann, X_test_ann, y_train_ann, y_test_ann = data['ann']
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = data['seq']
    class_weights = data['class_weights']
    
    num_classes = y_train_ann.shape[1]

    # Models Registry
    models = [
        (build_lstm_model(X_train_seq.shape[1:], num_classes), X_train_seq, y_train_seq, X_test_seq, y_test_seq),
        (build_ann_model(X_train_ann.shape[1:], num_classes), X_train_ann, y_train_ann, X_test_ann, y_test_ann),
        (build_gru_model(X_train_seq.shape[1:], num_classes), X_train_seq, y_train_seq, X_test_seq, y_test_seq),
        (build_hybrid_model(X_train_seq.shape[1:], num_classes), X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    ]

    for model, X_tr, y_tr, X_te, y_te in models:
        print(f"\n================ Training {model.name} ================")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        model.fit(
            X_tr, y_tr,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            class_weight=class_weights,
            verbose=1
        )

        loss, acc = model.evaluate(X_te, y_te, verbose=0)
        print(f"{model.name} Performance -> Loss: {loss:.4f} | Accuracy: {acc:.4f}")


if __name__ == '__main__':
    train_and_evaluate()