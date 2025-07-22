import random
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset

# BERT is a case-insensitive transformer model by Google pretrained on English data by self-supervised learning
model_name = "bert-base-uncased"

# 5 epochs gives us an avergae total loss of 0.25 which is ideal for two classifiers
epochs = 5

# Sentence Classification. Classes: "Informative", "Question", "Statement", "Exclamation"
class_labels = ['Informative', 'Question', 'Statement', 'Exclamation']
num_classes = len(class_labels)
class_id = {label: idx for idx, label in enumerate(class_labels)}

# Sentiment Analysis. Labels: "Positive", "Negative", "Neutral"
sentiment_labels = ['Positive', 'Negative', 'Neutral']
num_sentiments = len(sentiment_labels)
sentiment_id = {label: idx for idx, label in enumerate(sentiment_labels)}

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generated using gemini.ai. Prompt: "please generate a list of lists (to be used in a python program) that consists of 100 entries of [sentence, 
# classification ('Informative', 'Question', 'Statement', 'Exclamation'), sentiment ('Positive', 'Negative', 'Neutral')]"
training_data = [["The capital of France is Paris.", "Informative", "Neutral"],
          ["What is the population of New York City?", "Question", "Neutral"],
          ["The average temperature in London in July is around 20 degrees Celsius.", "Informative", "Neutral"],
          ["What is the main export of Brazil?", "Question", "Neutral"],
          ["The highest mountain in Nepal is Mount Everest.", "Informative", "Neutral"],
          ["What is the chemical formula for water?", "Question", "Neutral"],
          ["A key fact about photosynthesis is how plants make food.", "Informative", "Neutral"],
          ["Is it true that the Earth is round?", "Question", "Neutral"],
          ["Did you know that bees make honey?", "Question", "Neutral"],
          ["Have you ever wondered about black holes?", "Question", "Neutral"],
          ["I went to the store today.", "Statement", "Neutral"],
          ["I like pizza.", "Statement", "Positive"],
          ["The weather is sunny.", "Statement", "Positive"],
          ["This is a great idea.", "Statement", "Positive"],
          ["We should go to the park.", "Statement", "Positive"],
          ["It's a beautiful day.", "Statement", "Positive"],
          ["I think that it will rain.", "Statement", "Neutral"],
          ["The book was very interesting.", "Statement", "Positive"],
          ["The food was delicious.", "Statement", "Positive"],
          ["I am feeling happy.", "Statement", "Positive"],
          ["Wow, that was fast!", "Exclamation", "Positive"],
          ["Amazing, you did it!", "Exclamation", "Positive"],
          ["Incredible, the view!", "Exclamation", "Positive"],
          ["Fantastic, the party!", "Exclamation", "Positive"],
          ["Great, job!", "Exclamation", "Positive"],
          ["Excellent, news!", "Exclamation", "Positive"],
          ["Unbelievable, what happened!", "Exclamation", "Negative"],
          ["Look at that, the sunset!", "Exclamation", "Positive"],
          ["How wonderful, the music!", "Exclamation", "Positive"],
          ["What a surprise, to see you!", "Exclamation", "Positive"],
          ["The capital of Germany is Berlin.", "Informative", "Neutral"],
          ["What is the population of Tokyo?", "Question", "Neutral"],
          ["The average temperature in Moscow in January is around -10 degrees Celsius.", "Informative", "Neutral"],
          ["What is the main export of Saudi Arabia?", "Question", "Neutral"],
          ["The highest mountain in Argentina is Aconcagua.", "Informative", "Neutral"],
          ["What is the chemical formula for salt?", "Question", "Neutral"],
          ["A key fact about gravity is that it attracts objects with mass.", "Informative", "Neutral"],
          ["Is it true that the sky is blue?", "Question", "Neutral"],
          ["Did you know that dolphins are mammals?", "Question", "Neutral"],
          ["Have you ever wondered about the universe?", "Question", "Neutral"],
          ["I went to the library.", "Statement", "Neutral"],
          ["I like coffee.", "Statement", "Positive"],
          ["The weather is cold.", "Statement", "Negative"],
          ["This is a bad idea.", "Statement", "Negative"],
          ["We should stay home.", "Statement", "Negative"],
          ["It's a terrible day.", "Statement", "Negative"],
          ["I think that it will be sunny.", "Statement", "Neutral"],
          ["The movie was very boring.", "Statement", "Negative"],
          ["The soup was cold.", "Statement", "Negative"],
          ["I am feeling sad.", "Statement", "Negative"],
          ["Wow, that was slow!", "Exclamation", "Negative"],
          ["Amazing, you failed!", "Exclamation", "Negative"],
          ["Incredible, the damage!", "Exclamation", "Negative"],
          ["Fantastic, the mess!", "Exclamation", "Negative"],
          ["Great, loss!", "Exclamation", "Negative"],
          ["Excellent, failure!", "Exclamation", "Negative"],
          ["Unbelievable, the success!", "Exclamation", "Positive"],
          ["Look at that, the rain!", "Exclamation", "Negative"],
          ["How wonderful, the noise!", "Exclamation", "Negative"],
          ["What a surprise, to see him!", "Exclamation", "Neutral"],
          ["The capital of Italy is Rome.", "Informative", "Neutral"],
          ["What is the population of London?", "Question", "Neutral"],
          ["The average temperature in Sydney in July is around 15 degrees Celsius.", "Informative", "Neutral"],
          ["What is the main export of Japan?", "Question", "Neutral"],
          ["The highest mountain in Australia is Mount Kosciuszko.", "Informative", "Neutral"],
          ["What is the chemical formula for sugar?", "Question", "Neutral"],
          ["A key fact about the brain is that it controls the body.", "Informative", "Neutral"],
          ["Is it true that the sun is a star?", "Question", "Neutral"],
          ["Did you know that spiders are arachnids?", "Question", "Neutral"],
          ["Have you ever wondered about dreams?", "Question", "Neutral"],
          ["I went to the gym.", "Statement", "Neutral"],
          ["I like books.", "Statement", "Positive"],
          ["The weather is windy.", "Statement", "Neutral"],
          ["This is a complex issue.", "Statement", "Neutral"],
          ["We should consider this.", "Statement", "Neutral"],
          ["It's a lovely evening.", "Statement", "Positive"],
          ["I think that it will be cold.", "Statement", "Neutral"],
          ["The play was very good.", "Statement", "Positive"],
          ["The tea was hot.", "Statement", "Positive"],
          ["I am feeling tired.", "Statement", "Negative"],
          ["Wow, that was close!", "Exclamation", "Neutral"],
          ["Amazing, you recovered!", "Exclamation", "Positive"],
          ["Incredible, the speed!", "Exclamation", "Positive"],
          ["Fantastic, the view!", "Exclamation", "Positive"],
          ["Great, effort!", "Exclamation", "Positive"],
          ["Excellent, work!", "Exclamation", "Positive"],
          ["Unbelievable, the cost!", "Exclamation", "Negative"],
          ["Look at that, the stars!", "Exclamation", "Positive"],
          ["How wonderful, the silence!", "Exclamation", "Positive"],
          ["What a surprise, to see them!", "Exclamation", "Positive"],
          ["The capital of Canada is Ottawa.", "Informative", "Neutral"],
          ["What is the population of Mumbai?", "Question", "Neutral"],
          ["The average temperature in Buenos Aires in July is around 10 degrees Celsius.", "Informative", "Neutral"],
          ["What is the main export of China?", "Question", "Neutral"],
          ["The highest mountain in Africa is Mount Kilimanjaro.", "Informative", "Neutral"],
          ["What is the chemical formula for oxygen?", "Question", "Neutral"],
          ["A key fact about the heart is that it pumps blood.", "Informative", "Neutral"],
          ["Is it true that the Earth is flat?", "Question", "Negative"],
          ["Did you know that bats are mammals?", "Question", "Neutral"],
          ["Have you ever wondered about time travel?", "Question", "Neutral"],
          ["I went to the beach.", "Statement", "Positive"],
          ["I like music.", "Statement", "Positive"],
          ["The weather is rainy.", "Statement", "Negative"],
          ["This is a tough challenge.", "Statement", "Negative"],
          ["We should try harder.", "Statement", "Neutral"],
          ["It's a perfect morning.", "Statement", "Positive"],
          ["I think that it will be fine.", "Statement", "Positive"],
          ["The concert was amazing.", "Statement", "Positive"],
          ["The coffee was strong.", "Statement", "Positive"],
          ["I am feeling great.", "Statement", "Positive"],
          ["Wow, that was loud!", "Exclamation", "Negative"],
          ["Amazing, you survived!", "Exclamation", "Positive"],
          ["Incredible, the power!", "Exclamation", "Positive"],
          ["Fantastic, the show!", "Exclamation", "Positive"],
          ["Great, save!", "Exclamation", "Positive"],
          ["Excellent, timing!", "Exclamation", "Positive"],
          ["Unbelievable, the damage!", "Exclamation", "Negative"],
          ["Look at that, the moon!", "Exclamation", "Positive"],
          ["How wonderful, the peace!", "Exclamation", "Positive"],
          ["What a surprise, to see you here!", "Exclamation", "Positive"]]


"""
Defining the neural network model class. Returns embeddings and logits for classifiers
1. BERT is a case-insensitive transformer model by Google pretrained on English data by self-supervised learning. Using BERT's pooler_output 
([CLS] token embedding) as the sentence representation. The hidden state of the [CLS] token is often a better  representation of the entire 
sentence for classification tasks.
2. In order to support multitask learning, I added two torch.nn.Linear() functions with respective number of output classes and passed the embeddings
to the function to obtain logits
"""
class SentenceClassifier(torch.nn.Module):
  def __init__(self, base_model_name, num_classes, num_sentiments):
    super(SentenceClassifier, self).__init__()
    self.transformer = AutoModel.from_pretrained(base_model_name)
        
    self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, num_classes)
    self.sentiment = torch.nn.Linear(self.transformer.config.hidden_size, num_sentiments)

  def forward(self, input_ids, attention_mask):
    output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    sentence_embeddings = output.pooler_output
    class_logits = self.classifier(sentence_embeddings)
    sentiment_logits = self.sentiment(sentence_embeddings)
    return sentence_embeddings, class_logits, sentiment_logits

"""
Trains the model on the given data as it uses (stoichastic training)
"""
def train(model, data):
  
  print('Training...')

  # For richer datasets, full fine-tuning often yields better results
  for param in model.transformer.parameters():
    param.requires_grad = True

  model.train()

  # Only optimize parameters with requires_grad=True, this ensures that only the trainable parameters are updated
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
  class_criterion = torch.nn.CrossEntropyLoss()
  sentiment_criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(epochs):
    # data is shuffled to prevent overfitting
    random.shuffle(data)
    total_loss = 0.0
    count = 0
    for sentence, classification, sentiment in data:

      inputs = tokenizer(sentence, return_tensors="pt", padding='max_length',
                          truncation=True, max_length=32).to(device)
      class_tensor = torch.tensor([class_id[classification]]).to(device)
      sentiment_tensor = torch.tensor([sentiment_id[sentiment]]).to(device)
    
      embeddings, class_logits, sentiment_logits = model(inputs["input_ids"], inputs["attention_mask"])
      lossA = class_criterion(class_logits, class_tensor)
      lossB = sentiment_criterion(sentiment_logits, sentiment_tensor)
      loss = lossA + lossB
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      count += 1

    print(f"Epoch {epoch+1} | Loss: {total_loss/count:.4f}")


"""
Accepts a sentence, embeds it, and predcits the class and sentiment
"""
def predict(model, sentences):

  model.eval()
  with torch.no_grad():
    encoded = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    embeddings, class_logits, sentiment_logits = model(encoded["input_ids"], encoded["attention_mask"])
    predicted_class_ids = torch.argmax(class_logits, dim=1)
    predicted_class_labels = [class_labels[i] for i in predicted_class_ids]
    predicted_sentiment_ids = torch.argmax(sentiment_logits, dim=1)
    predicted_sentiment_labels = [sentiment_labels[i] for i in predicted_sentiment_ids]

    print(f"Shape of embeddings: {embeddings.shape}")
    
    # embeddings are of length 768. fixed size of 20 is printed out
    for i, embedding in enumerate(embeddings):
      print(f"Sentence: '{sentences[i] if type(sentences) == list else sentences}'")
      print(f"Embedding: {embedding[:20]}")
      print(f"Classification: '{predicted_class_labels[i]}'")
      print(f"Sentiment:'{predicted_sentiment_labels[i]}'")
      print("-" * 20)


"""
classifies input sentences using a basic neural network model trained on labeled examples
"""
def main():

  model = SentenceClassifier(model_name, num_classes, num_sentiments).to(device)

  train(model, training_data)
  
  sentences = ['I love LLMs!', 
               'Is this a sentence classifier?', 
               'LLMs are typically built on a type of neural network called a Transformer.', 
               'The weather outside is dreary']

  predict(model, sentences)
  
if __name__ =='__main__':
    main()
          
