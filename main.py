from langchain.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from datetime import datetime, date
from bs4 import BeautifulSoup
import feedparser
import requests

today = date.today()

prompt_template = """Je veux que tu me résumes cette article. Mais attention, il y a beaucoup de section qui n'ont rien à voir avec l'article.
              Je veux donc que tu te fies au titre pour savoir si c'est pertinent de résumer la section ou non.
              Si tu penses que la section n'est pas pertinente (par exemple les auteurs, ou des pubs), tu peux la sauter.
              ```{text}```
  """

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

urls = [
    "https://journalducoin.com/feed/", #Lien français
    "https://cryptoast.fr/feed/" #Lien français
]

llm = Ollama(model="openchat:7b-v3.5-q4_1")
chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

# fake user agent to avoid 403 error
headers = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}

#Récupère uniquement les articles d'aujourd'hui
def getTodayArticles(articles):
  return [article for article in articles if datetime.strptime(article.get("published"), "%a, %d %b %Y %H:%M:%S %z").date() == today]

# Cette fonction permet de récupérer les données d'un article, l'enregistre dans un fichier dans le dossier courant et renvoie le résumé de l'article
def extractContent(url):
    try:
        response = requests.get(url, headers=headers)
        bf = BeautifulSoup(response.content, 'html.parser')
        paragraphs = bf.find('div', class_='article-content').find_all('p')

        if("journalducoin" in url):
          paragraphs = bf.find('div', class_='content').find_all('p')

        body = ' '.join([p.get_text() for p in paragraphs])
        file_path = './' + url.split('/',5)[-2] + ".txt"

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(body)

        loader = TextLoader(file_path, encoding = 'utf-8')
        doc = loader.load()
        result = chain.invoke(doc)

        return result["output_text"]
    except Exception as e:
        return "Erreur lors de l'extraction du contenu : " + str(e)

def displayArticlesResumeFromUrls(urls):
  articles = []
  for url in urls:
    feed = feedparser.parse(url)
    today_articles = getTodayArticles(feed.entries)
    for entry in today_articles:
        print("\n\n\n\n\n")
        resume = extractContent(entry.link)

        output_text = "Titre: \t " + entry.title + "\n" + resume + "\n" + "Lien de l'article : " + entry.link

        print(output_text)
        articles.append(entry)
  print("Voici ci-dessus le résumé des " + str(len(articles)) + " actus blockchain du jour")
  return articles

try:
  displayArticlesResumeFromUrls(urls)
except Exception as e:
  print("Il y a eu une erreur lors de l'exécution du programme", e)
