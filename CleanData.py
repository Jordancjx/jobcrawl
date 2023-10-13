import csv
import pandas as pd
import numpy as np
import os
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib
import collections
import codecs

import pyLDAvis.gensim
import pickle
import pyLDAvis
from pprint import pprint
import gensim.corpora as corpora
import en_core_web_sm


def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

#
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

def total_tokens(text):
    n = WordPunctTokenizer().tokenize(text)
    return collections.Counter(n), len(n)

def make_df(counter, size):
    absolute_frequency = np.array([el[1] for el in counter])
    relative_frequency = absolute_frequency / size
    dfidk = pd.DataFrame(data=np.array([absolute_frequency, relative_frequency]).T, index=[el[0] for el in counter],
                      columns=["Absolute frequency", "Relative frequency"])
    dfidk.index.name = "Most common words"
    return dfidk

#csv stuff
csvFileName = "indeed_jobs_processed.csv"
csvHeading = ['jobTitle', 'jobCompany', 'jobLocation', 'jobMeta', 'jobContent', 'url']

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline="") as newcsv:
        writer = csv.DictWriter(newcsv, fieldnames=csvHeading)
        writer.writeheader()

os.chdir('...')
df = pd.read_csv(r'C:\Users\jiaen\Desktop\indeed_jobs.csv')
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

#remove meta data columns
df = df.drop(columns=['jobMeta'])

#remove punctuation/lowercasing
#df['jobContent'] = \

csvFileName2 = "output27.csv"
csvHeading2 = ['jobTitle', 'Skillsets', 'jobCompany', 'jobLocation', 'url']

if not os.path.exists(csvFileName2):
    with open(csvFileName2, 'w', newline="") as newcsv:
        writer = csv.DictWriter(newcsv, fieldnames=csvHeading2)
        writer.writeheader()

for i in range(0, len(df['jobContent'])):
    # skillsets = ['python', 'java', 'html', 'css']
    # filtered_skillsets = []
    # for j in range(0, len(df['jobContent'][i])):
    #     if df['jobContent'][i][j] in skillsets:
    #         filtered_skillsets.append(df['jobContent'][i][j])
    #
    # print(filtered_skillsets)

    text = ''.join(df['jobContent'][i]).lower()
    description_lines = text.split('\n')

    # Flag to indicate when to start and stop capturing the "Requirements" paragraph
    start_capturing = False
    requirements_paragraph = []

    # Iterate through the lines to capture the "Requirements" paragraph
    for line in description_lines:
        # print(description_lines)
        if "requirements" in line or 'skillsets' in line or "expertise" in line or 'qualifications' in line or 'required' in line or 'skills' in line or 'role' in line or 'responsibilities' in line or 'competencies' in line or 'competency' in line or 'bring' in line or 'proficiency' in line or 'skills' in line or 'expected' in line or 'experience' in line or 'prerequisites' in line or 'pre-requisites' in line or 'essential' in line or 'knowledge' in line:
            start_capturing = True
        elif 'job type' in line or 'salary' in line or 'resume' in line or 'interested' in line or 'tiktok' in line or 'invite' in line or 'team' in line or 'benefits' in line:
            break
        elif start_capturing and line.strip():  # If we have started capturing and the line is not empty
            requirements_paragraph.append(line)
        elif start_capturing and not line.strip():  # If we have started capturing and the line is empty, stop capturing
            break

    # Combine the captured lines to form the "Requirements" paragraph
    requirements_text = '\n'.join(requirements_paragraph)

    reviewDictEntry = {
            'jobTitle': df['jobTitle'][i],
            'Skillsets' : requirements_text,
            'jobCompany' : df['jobCompany'][i],
            'jobLocation' : df['jobLocation'][i],
            'url' : df['jobContent'][i]
        }
# print(reviewDictEntry)
    with open(csvFileName2, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=csvHeading2)
            w.writerow(reviewDictEntry)




#remove duplicates
drop_duplicates = df.drop_duplicates()

for i in range(0, len(df['jobContent'])):
    RE = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
    df['jobContent'][i] = RE.sub('', df['jobContent'][i])
    df['jobContent'][i] = df['jobContent'][i].strip()  # remove whitespaces
    df['jobContent'][i] = df['jobContent'][i].replace(r'[\u4e00-\u9fff\^\w\s\]|_', '')
    df['jobContent'][i] = re.sub('[^A-Za-z0-9]+', ' ', df['jobContent'][i])
    df['jobContent'][i] = df['jobContent'][i].replace('\n', ' ')
    df['jobContent'].map(lambda x: re.sub('[,\.!?]', '', x))

    reviewDictEntry = {
        'jobTitle': df['jobTitle'][i],
        'jobCompany': df['jobCompany'][i],
        'jobLocation': df['jobLocation'][i],
        'url': df['url'][i],
        'jobContent': df['jobContent'][i]
    }
    # print(reviewDictEntry)

    with open(csvFileName, 'a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csvHeading)
        w.writerow(reviewDictEntry)

stop_words = set(stopwords.words('english'))
df['jobContent'] = remove_stopwords(df['jobContent'])
#standardize the joblocation
df['jobLocation'] = df['jobLocation'].str.strip() #remove whitespaces
df['jobLocation'] = df['jobLocation'].apply(lambda x:'Singapore' if 'Singapore' in x else 'Not Found')
df['jobLocation'] = 'Singapore'

# #Join different jobcontent together
# print(df['jobContent'][0])
# df['jobContent'] = list(sent_to_words(df['jobContent']))
long_string = ' '.join(map(str, df["jobContent"]))
#print(long_string)



# #create wordcloud object
# wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# #generate the wordcloud
# wordcloud.generate(long_string)
# image = wordcloud.to_image()
# image.save('wordcloud.png')

df = df.drop(columns=['jobCompany', 'jobLocation', 'url'], axis=1)



lemmatizer = WordNetLemmatizer()

text1_counter, text1_size = total_tokens(re.sub('\W+\s*', ' ', long_string))
all_df = make_df(text1_counter.most_common(1500), 1)
x = all_df.index.values
# print(all_df)
df_data = []
for i in range(0, len(x)+1):
    try:
        df_c = text1_counter.get(x[i], 0) / text1_size
        df_data.append([df_c])
        dist_df = pd.DataFrame(data=df_data, index=x, columns=["relative frequency"])
        dist_df.index.name = "Most Common Words"
        # dist_df.sort_values("Relative Frequency Difference", ascending=False, inplace=True)
        # print(len(word), word)
        dist_df.head(10)
    except:
        print('\n')


    #print(dist_df)
# dist_df.to_csv("output4.csv")

























"""TOPIC CLASSIFICATION yes"""
#df['processed_description'] = df['jobContent'].apply(preprocess_text)
#
data = df.jobContent.values.tolist()
#
data_words = list(sent_to_words(data))
# # remove stop words
data_words = remove_stopwords(data_words)

#
# # Create Dictionary
# id2word = corpora.Dictionary(data_words)
# # Create Corpus
# texts = data_words
# import spacy
# nlp = spacy.load("en_core_web_sm")
# skillsets = []
#
# skill_keywords = [
#     "software development",
#     "large-scale data analytics",
#     "AI",
#     "programming",
#     "algorithms",
# ]
#
#
# def extracttext(doc):
#     for sentence in doc.sents:
#         for token in sentence:
#             if token.text.lower() in ["python", "java", "html", "css", "javascript"]:
#                 skillsets.append(token)
#
# # print(skillsets)
#
#
# for i in range(0, len(df['jobContent'])):
#     for sentence in df['jobContent']:
#         for token in sentence:
#             print(token)
#             if token.lower() in ["python", "java", "html", "css", "javascript"]:
#                 skillsets.append(sentence)
#
# print(skillsets)
# # # Print the extracted skillsets
# for idx, skillset in enumerate(skillsets, start=1):
#     print(f"Skillset {idx}: {skillset}")

#Term Document Frequency
# corpus = [id2word.doc2bow(text) for text in texts]
# #View
# print(corpus[:1][0][:30])
# corpus = []
# text = list(df['jobContent'])
# for i in range(len(text)):
#     r = ''.join(text[i])
#     corpus.append(r)
# # print(corpus)
# # check missing values
# df.isna().sum()
# # check data shape
# df.shape
# # print(df['jobTitle'])
# df['jobContent'].value_counts(normalize = True).plot.bar()
#
# X = df['jobContent']
# y = df['jobTitle']
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
# print('Training Data :', X_train.shape)
# print('Testing Data : ', X_test.shape)
#
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer()
# X_train_cv = cv.fit_transform(X_train)
# print(X_train_cv.shape)
# # print(corpus)
#
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(X_train_cv, y_train)
# # transform X_test using CV
# X_test_cv = cv.transform(X_test)
# # generate predictions
# predictions = lr.predict(X_test_cv)
# # print(predictions)
# print(y_test)
# print(predictions)

# from sklearn import metrics
# df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham', 'spam'], columns=['ham', 'spam'])









"""TOPIC MODELING TOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELINGTOPIC MODELING"""
# #number of topics
# num_topics = 100
# #Build LDA model
#
# lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
# #Print the Keyword in the 10 topics
#
# lda = gensim.models.ldamodel.LdaModel
# lda_model = lda(corpus, num_topics=num_topics, id2word=id2word, passes=1, random_state=0, eval_every=None)
# print(lda_model.print_topics())
#
# for idx, topic in lda_model.show_topics(formatted=False, num_words= 30):
#     print('Topic: {} \nWords: {}'.format(idx, '|'.join([w[0] for w in topic])))
#
# doc_lda = lda_model[corpus]
# print('test')
#
# df.to_csv(r'C:\Users\jiaen\Desktop\job_descriptions_with_topics.csv', index=False)
#

# # pyLDAvis.enable_notebook()
#
# LDAvis_data_filepath = os.path.join('./results')
# if 1 == 1:
#     LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#     with open(LDAvis_data_filepath, 'wb') as f:
#         pickle.dump(LDAvis_prepared, f)
# # load the pre-prepared pyLDAvis data from disk
# with open(LDAvis_data_filepath, 'rb') as f:
#     LDAvis_prepared = pickle.load(f)
#
# pyLDAvis.save_html(LDAvis_prepared, 'idk2.html')
# LDAvis_prepared







