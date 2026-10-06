# Generative AI with LSTM - Text Generation
# By Vijay Kumar Gupta
import re, urllib.request, numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "shakespeare.txt")
text = open("shakespeare.txt",encoding="utf-8").read().lower()
text = re.sub(r'[^a-z\s]', '', text)
text = re.sub(r'\s+', ' ', text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index)+1

seq_len=50
tokens=text.split()
seqs=[tokens[i-seq_len:i+1] for i in range(seq_len,len(tokens))]
enc=np.array(tokenizer.texts_to_sequences(seqs))
X=enc[:,:-1]
y=tf.keras.utils.to_categorical(enc[:,-1], num_classes=vocab_size)

split=int(0.9*len(X))
X_train,X_val=X[:split],X[split:]
y_train,y_val=y[:split],y[split:]

# 2. Model
model=Sequential([
  Embedding(vocab_size,100,input_length=seq_len-1),
  LSTM(150,return_sequences=True),
  Dropout(0.2),
  LSTM(150),
  Dropout(0.2),
  Dense(vocab_size,activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# 3. Training
model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=20,batch_size=128,
  callbacks=[EarlyStopping(patience=3,restore_best_weights=True),
  ModelCheckpoint('best_model.h5',save_best_only=True)],verbose=1)

# 4. Generation
def generate(seed,next_words=50):
  res=seed
  for _ in range(next_words):
    tl=tokenizer.texts_to_sequences([seed])[0]
    tl=pad_sequences([tl],maxlen=seq_len-1,truncating='pre')
    pred=np.argmax(model.predict(tl,verbose=0),axis=-1)[0]
    out=tokenizer.sequences_to_texts([[pred]])[0]
    seed+=" "+out; res+=" "+out
  return res

print(generate("the king said to his",30))
print(generate("to be or not to",30))
print(generate("love is a beautiful",30))