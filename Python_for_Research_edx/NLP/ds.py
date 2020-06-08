import os
from books import *
book_dir = "Books"


import pandas as pd
from books import *

# Creates an empty table with some columns
stats = pd.DataFrame(columns = ["language", "author", "title", "length", "unique"])

title_num = 1


for language in os.listdir(book_dir):
	for author in os.listdir(book_dir + "/" + language):
		for title in os.listdir(book_dir + "/" + language + "/" + author):
			inputfile = book_dir + "/" + language + "/" + author + "/" + title
			#print(inputfile)
			text = read_book(inputfile)
			(num_unique, counts) = word_stats(count_words(text))
			
			# Inserting the data into our table by columns
			stats.loc[title_num] = language, author.capitalize(), title.replace(".txt", ""), sum(counts), num_unique

			title_num += 1


print(stats.head(5))
#print(stats.tail(5))
print(stats.describe())

# ======== Just DS (?) ============
import matplotlib.pyplot as plt
'''
plt.figure()
plt.subplot(121)
plt.plot(stats.length, stats.unique, "ro")
plt.subplot(122)
plt.loglog(stats.length, stats.unique, "ro")
plt.show()
'''
plt.figure(figsize= (5, 5))
subset = stats[stats.language == "English"]
plt.loglog(subset.length, subset.unique, "o", label = "English", color = "crimson")

subset = stats[stats.language == "French"]
plt.loglog(subset.length, subset.unique, "o", label = "French", color = "forestgreen")

subset = stats[stats.language == "German"]
plt.loglog(subset.length, subset.unique, "o", label = "German", color = "orange")

subset = stats[stats.language == "Portuguese"]
plt.loglog(subset.length, subset.unique, "o", label = "Portuguese", color = "blueviolet")

plt.legend()
plt.xlabel("Book length")
plt.ylabel("Number or unique words")
plt.show()






# ======================== PANDAS ======================

#table = pd.DataFrame(columns = {"name", "age"})

#table.loc[1] = "James", 22 # Row 1 of a table
#table.loc[2] = "Jesss", 32

#print(table)