"""
corpus.py
---------
Drop-in replacement corpus for 01_bpe_tokenizer.py.

Paste any of the CORPUS_* variables below as the CORPUS in your script.
Each one is designed to teach you something different about BPE behavior.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS_DEFAULT — General English, good baseline to start with
# What to observe: common English subwords emerge (th, er, ing, ion, the, and)
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_DEFAULT = """
the cat sat on the mat
the cat ate the rat
the rat ran from the cat
a fat cat sat on a flat mat
cats are not rats and rats are not cats
the dog ran across the park in the morning
dogs and cats are common household pets
the rabbit hopped over the fence and into the garden
birds fly south in winter and north in summer
the fish swam deep in the cold river water
fish and chips is a popular meal in england
the sun rises in the east and sets in the west
morning light filters through the window and onto the floor
she walked to the market and bought bread and butter
he sat by the fire and read a book all evening
the children played in the garden until the sun went down
they laughed and ran and jumped and fell and laughed again
the old man fed the pigeons in the square every morning
a gentle wind blew through the trees and rustled the leaves
the rain fell softly on the rooftop through the long night
"""

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS_TECH — Software engineering and machine learning domain
# What to observe: 'def', 'return', 'model', 'data', 'train' merge early
# Try this after DEFAULT and compare which tokens are different
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_TECH = """
the model was trained on a large dataset of labeled examples
training a neural network requires a lot of data and compute
the loss function measures how wrong the model predictions are
gradient descent updates the weights to minimize the loss
backpropagation computes gradients through the network layers
the learning rate controls how fast the model learns from data
a high learning rate causes the model to overshoot the minimum
a low learning rate makes training very slow but more stable
the validation loss tells you if the model is overfitting
overfitting happens when the model memorizes training data
regularization techniques reduce overfitting in neural networks
dropout randomly disables neurons during training to prevent overfitting
batch normalization stabilizes training by normalizing layer inputs
the transformer architecture uses attention to process sequences
attention scores determine how much each token attends to others
the embedding layer converts tokens to dense vector representations
the tokenizer splits text into tokens before passing to the model
python is the most popular language for machine learning projects
numpy arrays are the foundation of numerical computing in python
pandas dataframes make it easy to load and clean tabular data
scikit learn provides simple apis for classic machine learning models
the pipeline chains preprocessing and model steps together cleanly
cross validation gives a more reliable estimate of model performance
hyperparameter tuning finds the best settings for your model
random search is faster than grid search for large hyperparameter spaces
feature engineering transforms raw data into useful model inputs
embeddings capture semantic meaning in dense vector representations
cosine similarity measures the angle between two embedding vectors
vector databases store and retrieve embeddings efficiently at scale
retrieval augmented generation combines search with language models
the context window limits how much text a model can process at once
fine tuning adapts a pretrained model to a specific downstream task
lora reduces the number of trainable parameters during fine tuning
the rank of the lora adapter controls how much the model can change
quantization reduces model size by using lower precision weights
inference is much faster than training for deployed language models
def train_model takes a dataset and returns a trained model object
return loss backward pass computes gradients for all parameters
class NeuralNetwork inherits from torch nn module base class
self attention computes queries keys and values from the same input
the forward method defines how data flows through the network layers
"""

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS_MIXED — English + Hindi + Kannada
# What to observe: English words get cleanly merged, Indian script words
# stay fragmented. Count tokens per word across languages.
# THIS is why non-English text costs more API tokens.
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_MIXED = """
the cat is sitting on the mat
बिल्ली चटाई पर बैठी है
ಬೆಕ್ಕು ಚಾಪೆಯ ಮೇಲೆ ಕುಳಿತಿದೆ
the dog runs in the park every morning
कुत्ता हर सुबह पार्क में दौड़ता है
ನಾಯಿ ಪ್ರತಿ ಬೆಳಿಗ್ಗೆ ಉದ್ಯಾನದಲ್ಲಿ ಓಡುತ್ತದೆ
machine learning is transforming every industry
मशीन लर्निंग हर उद्योग को बदल रही है
ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಪ್ರತಿಯೊಂದು ಉದ್ಯಮವನ್ನು ಬದಲಾಯಿಸುತ್ತಿದೆ
bangalore is a city in the state of karnataka
ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ
बेंगलुरु कर्नाटक राज्य की राजधानी है
she studied artificial intelligence at the university
उसने विश्वविद्यालय में आर्टिफिशियल इंटेलिजेंस पढ़ी
ಅವಳು ವಿಶ್ವವಿದ್ಯಾಲಯದಲ್ಲಿ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಅಧ್ಯಯನ ಮಾಡಿದಳು
the food was delicious and the service was excellent
खाना बहुत स्वादिष्ट था और सेवा उत्कृष्ट थी
ಆಹಾರ ತುಂಬಾ ರುಚಿಕರವಾಗಿತ್ತು ಮತ್ತು ಸೇವೆ ಅತ್ಯುತ್ತಮವಾಗಿತ್ತು
"""

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS_BIASED — One domain repeated heavily to simulate biased training data
# What to observe: "customer", "order", "product", "price" merge very early
# because their subwords dominate pair frequencies.
# Real lesson: LLM tokenizers trained on web crawl data are biased toward
# whatever was most common in that crawl (English, tech, news).
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_BIASED = """
the customer placed an order for the product online
the product was delivered to the customer address
customer service resolved the order issue quickly
the order was cancelled because the product was out of stock
the customer requested a refund for the damaged product
product reviews help other customers make better decisions
the customer rated the product five stars after delivery
order confirmation emails are sent to the customer immediately
the product price includes shipping and handling charges
customers can track their order status on the website
the return policy allows customers to return products within thirty days
customer satisfaction is the most important product metric
product inventory must be updated after every customer order
the customer support team handles product complaints daily
order processing takes one to two business days for most products
customers who order premium products receive priority shipping
the product catalog is updated every week with new customer favorites
customer loyalty programs reward repeat orders with product discounts
the product description must accurately reflect what the customer receives
order history shows every product the customer has ever purchased
"""

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS_CODE — Python source code as corpus
# What to observe: 'def', 'self', 'return', 'import', 'class' merge early.
# Indentation and punctuation stay fragmented.
# This is what happens inside a code-trained tokenizer like StarCoder.
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_CODE = """
def get_embeddings sentences model return model encode sentences
def cosine similarity a b return dot product a b divided by norm a times norm b
class TokenizerBPE def init self corpus num merges
self vocab self build vocab corpus
self merge rules list
def train self for step in range self num merges
pairs self get pair frequencies self vocab
if not pairs break
best pair max pairs key pairs get
self merge rules append best pair
self vocab self merge pair best pair self vocab
def tokenize self text tokens list
for word in text split symbols list word plus end marker
for pair in self merge rules
symbols self apply merge symbols pair
tokens append symbols
return tokens
import numpy as np
from collections import defaultdict Counter
from sklearn metrics pairwise import cosine similarity
from sentence transformers import SentenceTransformer
model SentenceTransformer all MiniLM L6 v2
embeddings model encode sentences normalize embeddings True
similarity matrix cosine similarity embeddings
top pairs list
for i in range len sentences
for j in range i plus one len sentences
top pairs append similarity matrix i j i j
top pairs sort reverse True
for score i j in top pairs five
print score sentences i sentences j
"""

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS_ECOMMERCE — Close to Saks product copy domain
# What to observe: luxury fashion vocabulary subwords emerge.
# 'leather', 'cotton', 'embroidered', 'collection' will develop shared subwords.
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_ECOMMERCE = """
this hand stitched leather handbag is crafted from full grain italian leather
the embroidered silk blouse features delicate floral patterns on the neckline
cashmere sweaters are made from the finest goat wool sourced from mongolia
this limited edition collection was designed by the award winning creative director
the tailored wool blazer has a structured silhouette and satin lined interior
genuine leather oxford shoes with a burnished toe cap and leather sole
the pleated midi skirt is made from lightweight crepe fabric in ivory
hand embroidered evening gown with crystal embellishments along the bodice
the quilted leather crossbody bag features gold tone hardware and chain strap
fine merino wool turtleneck in a relaxed fit with ribbed cuffs and hem
the printed silk scarf can be worn around the neck or tied to a handbag
vegetable tanned leather belt with a brushed silver rectangular buckle
the trench coat is cut from water resistant cotton gabardine fabric
beaded clutch bag with a magnetic snap closure and removable chain strap
the linen shirt dress has a relaxed fit with a self tie waist belt
patent leather pumps with a pointed toe and a slender stiletto heel
the cashmere cardigan features mother of pearl buttons and patch pockets
embossed leather wallet with card slots and a zip around coin compartment
the pleated palazzo trousers are crafted from flowing viscose fabric
hand painted ceramic buttons on a double breasted wool coat in camel
"""

# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE
# ─────────────────────────────────────────────────────────────────────────────
# In 01_bpe_tokenizer.py, replace the CORPUS variable:
#
#   from corpus import CORPUS_TECH      # or any variant
#   CORPUS = CORPUS_TECH
#
# Or paste one directly as the string value of CORPUS.
#
# Experiment sequence:
# 1. Run with CORPUS_DEFAULT    → baseline, see common English subwords
# 2. Run with CORPUS_TECH       → compare which new tokens appear
# 3. Run with CORPUS_BIASED     → watch 'customer'/'order' dominate early merges
# 4. Run with CORPUS_MIXED      → see token cost explosion for Kannada/Hindi
# 5. Run with CORPUS_ECOMMERCE  → relevant to your Saks work, see fashion vocab