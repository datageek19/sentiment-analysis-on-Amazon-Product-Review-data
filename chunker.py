import nltk

class Chunker():
	def __init__(self, n):
		# Load the corpus to train the chunk tagger
		conll = nltk.corpus.conll2000.chunked_sents()
		treebank = nltk.corpus.treebank_chunk.chunked_sents()
		data = conll + treebank

		chunks = [ nltk.chunk.tree2conlltags(tree) for tree in data ]
		chunks = [ [(tag[1], tag[2]) for tag in tags] for tags in chunks ]

		train_chunks = chunks#[:3000]
		#test_chunks = chunks[3000:]

		# Train the chunk tagger
		self.chunk_tagger = None
		if n == "Unigram":
			self.chunk_tagger = nltk.tag.UnigramTagger(train_chunks)
		elif n == "Bigram":
			self.chunk_tagger = nltk.tag.BigramTagger(train_chunks)
		elif n == "Trigram":
			self.chunk_tagger = nltk.tag.TrigramTagger(train_chunks)
		else:
			chunker = nltk.tag.UnigramTagger(train_chunks)
			chunker = nltk.tag.BigramTagger(train_chunks, backoff=chunker)
			chunker = nltk.tag.TrigramTagger(train_chunks, backoff=chunker)
			self.chunk_tagger = chunker
			#print('accuracy:', chunker.evaluate(test_chunks))

	def parseTree(self, tokens):
		# Gather the words, tags, and chunks
		words = [w for (w,t) in tokens]
		tags = [t for (w,t) in tokens]
		chunks = self.chunk_tagger.tag(tags)

		# Sanity check
		assert len(words) == len(tags)
		assert len(words) == len(chunks)

		# Build the parse tree
		l = []
		for i in range(0,len(words)):
			if chunks[i][1]:
				l.append( ' '.join([words[i], tags[i], chunks[i][1]]) )

		return nltk.chunk.conllstr2tree('\n'.join(l))

def chunkLabels(tree):
	l = []
	for t in tree.subtrees():
		if t.label() != 'S':
			#print(t.label())
			l.append(t.label())
	#quit()
	return ' '.join(l)


