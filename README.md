# sentiment-analysis-llm
Fine-tuned DistilBERT model for sentiment analysis


Hi! This is a simple project I built while learning about AI and Large Language Models.

In this project, I fine-tuned a pre-trained model (DistilBERT) to understand whether a sentence is **positive or negative**.



## 💡 Why I made this

I recently started exploring NLP and LLMs, and I wanted to try something practical instead of just theory.  
So I built this project to understand how real-world AI models are trained and used.

---

## 🧠 What this project does

- Takes a sentence as input
- Predicts if it's:
  - Positive 😊
  - Negative 😞

Example:

"This is amazing!" → POSITIVE  
"This is the worst experience ever." → NEGATIVE  

---

## ⚙️ Tech I used

- Python
- Hugging Face Transformers
- PyTorch
- Datasets library

---

## 🚀 How to run this project

1. Clone the repo:
   git clone https://github.com/pranjalbakliwal09/sentiment-analysis-llm.git

cd sentiment-analysis-llm

2. Create a virtual environment:
   python -m venv venv
venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the training:
   python train.py

---

## 📁 Project files

- `train.py` → main training code  
- `sentiment-model/` → saved trained model  
- `requirements.txt` → dependencies  

---

## 📸 Sample Output

<img width="2136" height="1390" alt="image" src="https://github.com/user-attachments/assets/381287e1-3ac6-46b4-8199-dd8f9361b943" />
<img width="1592" height="946" alt="image" src="https://github.com/user-attachments/assets/293ffb76-f275-4175-ad64-be500b121d81" />


Example:
Input: I love this → POSITIVE  
Input: I hate this → NEGATIVE  

---

## 🧪 What I learned

- How fine-tuning works  
- Using pre-trained models instead of building from scratch  
- Basics of Hugging Face Transformers  
- Handling real-world errors during development 😅  

---

## 🔧 Future improvements

- Train on a larger dataset  
- Deploy the model online  

---

## 🙌 Credits

This project uses open-source libraries from Hugging Face and PyTorch.

---

## 📌 Note

This is a beginner-level project and part of my learning journey in AI 🙂

---

## 📜 License

MIT License

