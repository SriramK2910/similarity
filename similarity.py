import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

"""Creating a class for finding similarity"""


class SentenceSimilarity:
    """Initializing variables and encoding with sentence transformer"""

    def __init__(self, var, var2):
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.var = model.encode(var)
        self.var2 = model.encode(var2)

    """Finding similarity between the given variables and input query and returning the maximum value of similarity"""

    def similarity(self):
        append_array = []
        for i in range(0, len(self.var)):
            cos_similar = cosine_similarity(
                self.var[i].reshape(1, -1), self.var2.reshape(1, -1)
            )
            append_array.append(cos_similar)
        max_similar = max(append_array)
        return max_similar


football = [
    "Association football, more commonly known as simply football or soccer,is a team sport played with a spherical ball between two teams of 11 players. It is played by approximately 250 million players in over 200 countries and dependencies",
    "Chelsea Football Club is an English professional football club based in Fulham, West London. Founded in 1905, the club competes in the Premier League, the top division of English football. Chelsea are among Englands most successful clubs",
    "Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England, that competes in the Premier League, the top flight of English football.",
]
cricket = [
    "Cricket is a bat-and-ball game played between two teams of eleven players on a field at the centre of which is a 22-yard (20-metre) pitch with a wicket at each end, each comprising two bails balanced on three stumps.",
    "The India men's national cricket team, also known as Team India and Men in Blue, is governed by the Board of Control for Cricket in India (BCCI), and is a Full Member of the International Cricket Council (ICC) with Test, One Day International (ODI) and Twenty20 International (T20I) status.",
]
climate_change = [
    "Climate change includes both human-induced global warming and its large-scale impacts on weather patterns. There have been previous periods of climate change, but the current changes are more rapid than any known events in Earth's history.",
    "The largest driver of warming is the emission of greenhouse gases, mainly carbon dioxide (CO2) and methane. Fossil fuel burning (coal, oil, and natural gas) for energy consumption is the main source of these emissions, with additional contributions from agriculture, deforestation, and the chemical reactions in certain manufacturing processes.",
]
finance = [
    "Indian equity benchmarks rose on Monday with teh 50-scrip Nifty index hitting the 18,000 mark for the first time ever, shrugging off weakness in other Asian markets. ",
    "Finance Minister Nirmala Sitharaman has embarked on a week-long US trip to attend the annual meet of the World Bank and IMF as well as G20 Finance Ministers and Central Bank Governors (FMCBG) meeting.",
    "TCS (Tata Consultancy Services) share price tanked nearly 7% on Monday morning as investors reacted to the quarterly results of the second most valuable listed company in India. TCS share price hit an intra-day low of Rs 3,660 per share, 6.9%.",
    "Finance is a term for matters regarding the management, creation, and study of money and investments.[1] [note 1] Specifically, it deals with the questions of how an individual, company or government acquires money – called capital in the context of a business – and how they spend or invest that money.",
]
input_query = input("Enter a sentence")
"""Giving the variables and input query in the class to find similarity"""

similar = SentenceSimilarity(football, input_query)
similarity_score_football = similar.similarity()
similar_1 = SentenceSimilarity(cricket, input_query)
similarity_score_cricket = similar_1.similarity()
similar_2 = SentenceSimilarity(climate_change, input_query)
similarity_score_climatechange = similar_2.similarity()
similar_3 = SentenceSimilarity(finance, input_query)
similarity_score_finance = similar_3.similarity()

"""Entering the key of the dictionary as the variable name whose similarity is to be calculates and the value is the similarity score"""
dict_of_scores = {
    "Football": similarity_score_football,
    "Cricket": similarity_score_cricket,
    "Climate Change": similarity_score_climatechange,
    "Finance": similarity_score_finance,
}

"""Taking maximum value out of the dictionary"""

max_values = max(dict_of_scores, key=dict_of_scores.get)
print(max_values)
