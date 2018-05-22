articles = open("data/train_article.txt", "r", encoding="utf-8").readlines()
abstracts = open("data/train_abstract.txt", "r", encoding="utf-8").readlines()
article_file = open("data/modified_train_article.txt", 'w', encoding="utf-8")
abstract_file = open("data/modified_train_abstract.txt", 'w', encoding="utf-8")

for i, (article, abstract) in enumerate(zip(articles, abstracts)):
    if article.strip() == '' or abstract.strip() == '':
        print("{} th line is empty".format(i))
        continue
    article_file.write(article)
    abstract_file.write(abstract)

abstract_file.close()
article_file.close()
