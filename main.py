import requests
from bs4 import BeautifulSoup
from gensim.summarization import summarize
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


# user_query = input("Enter Search Parameter---")
# no_of_keywords = int(input("How many key words per link?(1 to 7)---"))


def SearchableLinkGenerator(query):  # function to extract and process google links for scraping
    google_query = requests.get("https://www.google.com/search?q=" + query)  # searches web for respective query
    # wikipedia=requests.get("https://en.wikipedia.org/wiki/"+x)
    soup1 = BeautifulSoup(google_query.text, 'html.parser')  # creates bfulsoup object
    # print(soup1.prettify())
    # soup2=BeautifulSoup(wikihow.text,'html.parser')
    # soup3=BeautifulSoup(wikipedia.text,'html.parser')
    # print(google_results[:5])
    google_results = soup1.select('.kCrYT a')  # extracts links from soup object
    print("Getting Google links...")
    return google_results


def YoutubeLinkGenerator(query):  # function to generate youtube link for display
    query = query.replace(" ", "+")  # replaces space with + symbol to generate workable link
    youtube_query = requests.get("https://www.google.com/search?q=" + query + "+" + "video")
    soup2 = BeautifulSoup(youtube_query.text, 'html.parser')  # creates bfulsoup object
    # print(soup2.prettify())
    # print(("https://www.google.com/search?q=" + user_query + "+"+"video"))
    youtube_results = soup2.select('.kCrYT a')  # extracts links from soup object
    if "yout" in (youtube_results[0].get("href")):  # checks if link is a youtube link
        youtube_link = "https://google.com/" + (youtube_results[0].get("href"))
        return youtube_link
    else:
        return "No video"


def HeadlineAndSummaryGenerator(google_results, query):  # function to generate headline and summary
    print("Google links received...")
    google_link = []
    google_links = []
    headline_list = []
    summary_list = []
    article_list = []
    # google_links = set(google_links)
    # print(google_links)
    print("Scraping the links and generating Headline and Summary...")
    for index in range(0, 20):  # loop to traverse through list of links
        try:
            if "yout" not in (google_results[index].get("href")):  # checks for error inducing youtube link
                google_link.append(google_results[index].get("href"))
                if ("https://google.com/" + google_link[-1]) not in google_links:  # checks for unique link
                    google_links.append("https://google.com/" + google_link[-1])  # updates list of links
                # print(google_links)
            else:
                continue
        except:
            break
    for link in google_links:
        try:
            url = f"{link}"
            # print(url)
            page = requests.get(url).text  # gets text from page in link
            soup = BeautifulSoup(page, "html.parser")  # creates soup object
            # Get headline
            headline = soup.find('h1').get_text()  # finds headline
            # if headline_list.count(headline) > 0:
            # headline_list.append(soup.find('h1').get_text())
            # print(headline_list)
            p_tags = soup.find_all('p')  # Get text from all <p> tags.
            p_tags_text = [tag.get_text().strip() for tag in
                           p_tags]  # Get the text from each of the “p” tags and strip surrounding whitespace.
            sentence_list = [sentence for sentence in p_tags_text if
                             not '\n' in sentence]  # Filter out sentences that contain newline characters '\n'.
            sentence_list = [sentence for sentence in sentence_list if
                             '.' in sentence]  # Filter out sentences that don't contain periods.
            article = ' '.join(sentence_list)  # Combine list items into string.
            if "403" in headline:  # checks for bad gateway error in headline
                continue  # goes to next headline
            else:
                headline_list.append(headline)  # appends headline to headline list
            if headline_list.count(headline) < 2 and len(
                    article) > 200:  # checks for appropriate lengths of headline and article to be analysed
                article_list.append(article)  # add article to article list
                # print(count)
                # print(headline_list[-1])
                # print(len(article))
                # print(article_list)
                summary = summarize(article, ratio=0.4)  # summarizer function to summarize article
                if 19 < len(summary) < 4000:  # checks appropriate length of summary
                    summary_list.append(summary)  # add article summary to summary list
                    # print(len(summary))
                    # print(".")
                if len(summary) > 4000:  # checks appropriate length of summary
                    controlled_summary = summary[0:4000]  # edits length of summary to 4000 characters
                    last_fullstop = controlled_summary.rindex(
                        ".") + 1  # edits length of summary to last full stop within 4000 characters
                    summary_list.append(controlled_summary[:last_fullstop])  # add article summary to summary list
                    # print(len(summary))
        except:
            continue
    # print(headline_list[:5], summary_list[:5])
    return headline_list, article_list, summary_list


def TextListFeeder(article_list):  # function to feed list of articles to be summarized for nlp
    print("Scraping completed, Headline and Summary received...")
    to_predict_text_list = []
    for article in article_list:  # traversal of articles in articles list
        ideal_ratio = 1500 / len(article)  # tries to achieve ideal ratio of 1500 characters for nlp
        # print(len(article))
        if ideal_ratio > 1:  # checks for ideal ratio
            to_predict_text = summarize(article, ratio=1)  # summarizes article
        else:
            to_predict_text = summarize(article, ratio=round(ideal_ratio, 3))  # summarizes article
        if (len(to_predict_text)) > 1500:  # rechecks and tries to achieve ideal ratio
            # print(to_predict_text)
            re_run_value = 1500 / len(to_predict_text)  # rechecks and tries to achieve ideal ratio
            to_predict_text = summarize(to_predict_text, ratio=round(re_run_value, 3))  # summarizes article
            # print(f"double run--"+f"{re_run_value_list}")
            to_predict_text_list.append(to_predict_text)  # adds article to prediction list
        else:
            to_predict_text_list.append(to_predict_text[0:1500])  # adds article to prediction list
    print("Preparing Articles for Keyword-Extraction...")
    # print(to_predict_text_list[:5])
    return to_predict_text_list


def KeyWordExtractor(to_predict_text_list, keyword_count):  # function to extract keywords from articles
    value = 1  # count for number of articles
    last_article = 5  # control for article number
    keywords_list = []
    print("Articles prepared, starting Keyword-Extraction...")
    for item in to_predict_text_list[0:last_article]:  # traverses article summaries
        try:
            text = f"{item}"
            n_gram_range = (1, 2)  # sets the size of keywords to be extracted
            stop_words = "english"  # stopword language parameter
            # Extract candidate words/phrases
            count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(
                [text])  # fits the text into a vector model and generates list of n grams
            all_candidates = count.get_feature_names()  # gets name of extracted candidate n-grams
            # print(all_candidates[:10])
            nlp = spacy.load('en_core_web_sm')  # load spacy in english
            doc = nlp(text)
            noun_phrases = set(
                chunk.text.strip().lower() for chunk in doc.noun_chunks)  # deriving the phrases with the nouns
            nouns = set()  # extracting only nouns set
            for token in doc:
                if token.pos_ == "NOUN":  # checking for noun
                    nouns.add(token.text)
            all_nouns = nouns.union(noun_phrases)  # combining nouns and phrases with nouns
            candidates = list(filter(lambda candidate: candidate in all_nouns,
                                     all_candidates))  # filter out all the candidates to match with nouns set
            # print(candidates[:10])
            model_name = "distilroberta-base"  # using distilroberta model
            model = AutoModel.from_pretrained(model_name)  # model creation
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            candidate_tokens = tokenizer(candidates, padding=True, return_tensors="pt")  # generating candidate tokens
            candidate_embeddings = model(**candidate_tokens)[
                "pooler_output"]  # mapping candidate tokens to vector space
            # print(candidate_embeddings.shape)
            text_tokens = tokenizer([text], padding=True,
                                    return_tensors="pt")  # mapping input article text to vector space
            text_embedding = model(**text_tokens)["pooler_output"]  # mapping input article text to vector space
            # print(text_embedding.shape)
            candidate_embeddings = candidate_embeddings.detach().numpy()  # converting computed vectors to numpy arrays
            text_embedding = text_embedding.detach().numpy()  # converting computed vectors to numpy arrays

            top_k = int(keyword_count)  # alloting number of keywords required
            distances = cosine_similarity(text_embedding,
                                          candidate_embeddings)  # computing the distance according to cosine similarity
            keywords = [candidates[index] for index in
                        distances.argsort()[0][-top_k:]]  # sorting and storing the closest keywords to the text
            keywords_list.append(keywords)  # adding keywords to keywords list
            # print(keywords_list[-1])
            print(f"Keywords Extracted for Article {value}")
            value += 1
        except:
            print(f"Article {value} was not procured")
            value += 1
            last_article += 1  # to change last checked article
            continue
    # print(keywords_list[:5])
    return keywords_list


def Output(google_value, head_sum_value, final_keywords_list):  # final output function to be used for generating output
    google_value_list = []
    headline_value_list = []
    summary_value_list = []
    for i in range(0, 5):  # traverses for top 5 articles,can be modified to top n
        try:
            google_value_list.append("https://google.com/" + str((google_value[i]).get("href")))  # saves data in required list
            headline_value_list.append((head_sum_value[0][i]).replace('\n', ' '))  # saves data in required list
            summary_value_list.append(head_sum_value[2][i].replace("\n", " "))  # saves data in required list
        except:
            continue
    return google_value_list, headline_value_list, summary_value_list, final_keywords_list


# print(YoutubeLinkGenerator(user_query))

"""x = SearchableLinkGenerator(user_query)
y = HeadlineAndSummaryGenerator(x)
z = TextListFeeder(y[1])
final = KeyWordExtractor(z, no_of_keywords)
for i in range(0, 5):
    try:
        print("Link---" + "https://google.com/" + str((x[i]).get("href")))
        print("Headline---" + (y[0][i]).replace('\n', ' '))
        print("Summary---" + y[2][i])
        print("Keywords---" + str(final[i]))
    except:
        continue"""
