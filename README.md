# 🌐 Language Translator App (English → French)

A many-to-many encoder–decoder sequence model built with LSTM to translate English to French in real-time. Developed following the DataFlair tutorial “Language Translation with Machine Learning”.

---

## 🧠 Overview

- Implements a sequence-to-sequence (seq2seq) model with LSTMs.
- Uses **teacher forcing** during training for more stable convergence.
- Provides both training and GUI modules for interactive translation.

---

## 🚀 Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow ≥2.2 (Keras API)
- **Core Libraries**: numpy, sklearn, pickle
- **GUI**: tkinter (via `LangTransGui.py`)
- **Data**: English-French parallel sentences (`eng-french.txt`)

---

## 📂 Project Structure

```text
/
├── eng-french.txt          # Parallel corpus (training data)
├── langTraining.py         # Seq2seq model training
├── training_data.pkl       # Preprocessed training arrays
├── s2s/                    # Saved model weights, optimizer & metrics
├── LangTransGui.py         # GUI to load model and translate
└── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Saptha-Harsh/Language-Translation.git
   cd Language-Translation
   ```

2. **Install dependencies**  
   ```bash
   pip install tensorflow numpy scikit-learn pickle5
   ```

3. **Dataset**  
   Ensure `eng-french.txt` is present: each line contains `English_sentence<TAB>French_sentence`.

4. **Train the model**  
   ```bash
   python langTraining.py
   ```
   - Trains on ~10,000 sentence pairs (adjustable).
   - Saves model and preprocessing in `s2s/`.

5. **Launch the GUI translator**  
   ```bash
   python LangTransGui.py
   ```
   - Enter English text → click *Translate* → see French output.

---

## 🧩 How It Works

1. **Preprocessing**: Tokenize and vectorize input/output sentences into sequences.
2. **Encoder**: LSTM reads input English sentence.
3. **Decoder**: LSTM generates French translation, using teacher forcing during training.
4. **Teacher Forcing**: During training, the decoder gets true previous tokens for faster convergence.
5. **Inference**: GUI takes user input, tokenizes, and uses trained model to predict French output.

---

## 🛠️ Customization Tips

- **Increase dataset size**: Improve accuracy by using the full dataset.
- **Tweak hyperparameters**: Try different `batch_size`, `epochs`, or LSTM hidden dimensions.
- **Add attention**: Boost performance by integrating attention mechanisms.
- **Expand to other languages**: Substitute dataset, update tokenizers, retrain model.

---

## 📚 References

- DataFlair Project: https://data-flair.training/blogs/language-translation-machine-learning/
- TensorFlow Keras Seq2Seq Documentation
- Concepts of Teacher Forcing and LSTM-based Translation Models

---

## 🤝 Contributions & Contact

Contributions, suggestions, or improvements are welcome! Feel free to:
- ⭐ Star & fork the repo.
- 🐛 Report issues or suggest features.

---

**Enjoy experimenting & translating!**
