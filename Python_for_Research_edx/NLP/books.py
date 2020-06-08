import os

#path= 'Books_EngFr/English/shakespeare/Romeo and Juliet.txt'
#path1 = 'Books_GerPort/German/shakespeare/Romeo und Julia.txt'

def read_book(title_path):
	"""
	Read a book and return ir as a string.
	"""
	with open(title_path, "r", encoding= "utf8") as cf:
		text = cf.read()
		text = text.replace("\n", "").replace("\r", "")
		return text

		
#text = read_book(path)
#print(len(text))

'''
ind = text.find("What's in a name?")		
print(text[ind: ind+100])
'''

def word_stats(word_counts):
	"""Return number of unique words and their frequences"""
	num_unique = len(word_counts)
	counts = word_counts.values()
	return (num_unique, counts)


from collections import Counter
def count_words(text):
	""" Counts the worlds and returns the dicctionary with them """
	text = text.lower()
	skips = [".",",",";",":","'",'"']
	for ch in skips:
		text = text.replace(ch,"")

	world_counts = Counter(text.split(" "))
	return world_counts
'''
# English version
word_counts = count_words(text)
(num_unique, counts) = word_stats(word_counts)
print("Unique words: ",num_unique, "Lenght: ",sum(counts))

# German version
text1 = read_book(path1)
word_counts1 = count_words(text1)
(num_unique1, counts1) = word_stats(word_counts1)
print("Unique words: ",num_unique1, "Lenght: ",sum(counts1))
'''







