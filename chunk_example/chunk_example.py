import nltk
import chunker
import pandas as pd

df = pd.read_csv("consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
df = df[['reviews.rating', 'reviews.text', 'reviews.title']]
df = df[df["reviews.rating"].notnull()]
#df['reviews.text'] = df['reviews.text'].apply(str.split)
#df['reviews.text'] = df['reviews.text'].apply(nltk.pos_tag)


#ch = chunker.Chunker("Unigram")
#ch = chunker.Chunker("Bigram")
ch = chunker.Chunker("Trigram")

# Show the parse trees for the first 5 rows
for i in range(5):
	text = df['reviews.text'].iloc[i]
	tagged = nltk.pos_tag( text.split() )
	tree = ch.parse(tagged)
	print(tree)
	tree.draw()
