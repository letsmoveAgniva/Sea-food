Our chatbot mostly utilizes BARD and Langchain for question answering.BARD is used for 3 different purposes in our bot:

Queries related to seafood and seafood business
Providing description from Images
Reading from personal csv files We used beautiful soup for scrapping data from MPEDA and other websites and cleaned the text and made a pdf. Langchain is used to do question answering from this pdf Web scraping data from the MPEDA official website and multiple seafood export company websites, followed by text cleaning, preprocessing using NLTK & re library.
EfficientNet for classifying image,Flask as the main python framework for integration,News API for showing recent news related to seafood business,Google Firebase - creating database of login,smtplib for providing follow up emails with proper summary
