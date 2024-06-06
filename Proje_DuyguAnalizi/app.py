from flask import Flask, render_template, request
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import concurrent.futures

app = Flask(__name__)

model = load_model("model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_tokens = 100

def predict_sentiment(text):
    text_tokens = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_tokens, maxlen=max_tokens)
    prediction = model.predict(text_pad)[0][0]
    if prediction >= 0.5:
        sentiment = 'Olumlu 😀'
        probability = prediction * 100
    else:
        sentiment = 'Olumsuz 😞'
        probability = (1 - prediction) * 100
    return sentiment, probability

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def scrape_reviews(base_url, num_pages, site):
    tum_yorumlar = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for page_num in range(1, num_pages + 1):
            if site == 'sinefil':
                url = f"{base_url}/{page_num}"
            else:
                url = f"{base_url}?page={page_num}"
            tasks.append(fetch(session, url))
        
        pages = await asyncio.gather(*tasks)

        for page in pages:
            soup = BeautifulSoup(page, 'html.parser')
            if site == 'beyazperde':
                yorumlar = soup.find_all('div', class_='content-txt review-card-content')
                for yorum in yorumlar:
                    yorum_metni = yorum.get_text().strip()
                    if yorum_metni not in tum_yorumlar:
                        tum_yorumlar.append(yorum_metni)
            elif site == 'sinefil':
                yorumlar = soup.find_all('div', class_='readmore-300')
                for yorum in yorumlar:
                    yorum_metni = yorum['data-salt-content']
                    if yorum_metni not in tum_yorumlar:
                        tum_yorumlar.append(yorum_metni)
    
    return tum_yorumlar

def analyze_reviews(film_url, num_reviews, site):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tum_yorumlar = loop.run_until_complete(scrape_reviews(film_url, 25, site))
    actual_num_reviews = min(num_reviews, len(tum_yorumlar))
    review_data = []
    
    if num_reviews > len(tum_yorumlar):
        error_message = f"Girdiğiniz yorum sayısı sitede bulunan yorum sayısından fazla. Sitede bulunan yorum sayısı: {len(tum_yorumlar)}"
    else:
        error_message = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(predict_sentiment, tum_yorumlar[i]): i for i in range(actual_num_reviews)}
        results = {i: future.result() for future, i in future_to_index.items()}

    for i in range(actual_num_reviews):
        sentiment, probability = results[i]
        review_text = tum_yorumlar[i]
        short_text = review_text[:100] + '...' if len(review_text) > 100 else review_text
        review_data.append({
            "index": i + 1,
            "review_text": review_text,
            "short_text": short_text,
            "sentiment": sentiment,
            "probability": probability
        })

    return review_data, actual_num_reviews, error_message

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    review_data = []
    sentiment = None
    probability = None
    actual_num_reviews = 0
    film_url = ""
    site = ""
    
    if request.method == 'POST':
        if 'comment' in request.form:  
            comment = request.form['comment']
            sentiment, probability = predict_sentiment(comment)
        elif 'film_url' in request.form and 'num_reviews' in request.form: 
            film_url = request.form['film_url']
            num_reviews = int(request.form['num_reviews'])
            site = request.form['site']
            
            if film_url and num_reviews and site:
                try:
                    review_data, actual_num_reviews, error_message = analyze_reviews(film_url, num_reviews, site)
                except Exception as e:
                    error_message = str(e)
    
    return render_template('WebSide.html', 
                           reviews=review_data, 
                           sentiment=sentiment, 
                           probability=probability, 
                           error_message=error_message, 
                           film_url=film_url, 
                           actual_num_reviews=actual_num_reviews,
                           site=site)

if __name__ == '__main__':
    app.run(debug=True)
