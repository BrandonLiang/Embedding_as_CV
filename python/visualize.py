from PIL import Image
from transformers import BertModel, BertTokenizer
import torch
from sklearn.preprocessing import StandardScaler

def visualize(bert_model, sample_word = "snow", epoch = 1, location = "/Users/brandonliang/src/embedding_as_cv/image")
  bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  sample_tensor = torch.tensor([bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sample_word))])
  sample_word_embedding = bert_model(sample_tensor)[0].detach().view(-1).view(24,32).numpy()
  sample_word_embedding = StandardScaler().fit_transform(sample_word_embedding)
  #print(sample_word_embedding[0].shape, sample_word_embedding[1].shape)
  img = Image.fromarray(sample_word_embedding, mode = 'P')
  #print(sample_word_embedding)
  #print(img.size)
  size = 320, 240
  #img = img.thumbnail(size, Image.ANTIALIAS)
  img = img.resize(size, Image.ANTIALIAS)
  #print(img.size)
  img.save("{}/{}.png".format(location, epoch))

if __name__ == "__main__":
  visualize(sample_word = "snow")
