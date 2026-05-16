import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("Euroviisu-analyysi")

# 1. Lataus ja kertoimien suodatus
odds_df = pd.read_csv('betting_offices.csv')
mask = (odds_df['contest_round'] == 'final')
clean_odds = odds_df[mask].copy()

# 2. Ryhmitys ja prosenttien laskenta KAIKILLE (matematiikan takia)
summary = clean_odds.groupby(['year', 'country_name'])['betting_score'].mean().reset_index()
summary['prob'] = 1 / summary['betting_score']
summary['win_chance'] = summary.groupby('year')['prob'].transform(lambda x: (x / x.sum()) * 100).round(1)

# 3. Yhdistetään tuloksiin
results_df = pd.read_csv(
    'contestants.csv', 
    header=None, 
    usecols=[0, 1, 2, 3, 4, 5, 7], 
    names=['year', 'country_code', 'country_name', 'song', 'performer', 'place', 'running_order']
)

final_df = pd.merge(summary, results_df, on=['year', 'country_name'])
final_df = final_df.dropna(subset=['place']).drop_duplicates(subset=['year', 'country_name'])

# --- LISÄTÄÄN TÄMÄ RIVI: Poimitaan TOP 5 ennakkosuosikit kertoimien mukaan ---
top5_favorites = final_df.sort_values(['year', 'win_chance'], ascending=[True, False]).groupby('year').head(5).copy()

# 4. RAJAUS: Vain ne, jotka olivat lopulta TOP 5 sijoilla
top5_results = final_df[final_df['place'] <= 5].copy()
print(top5_results[['year', 'country_name', 'place', 'win_chance']])

# Pakotetaan vuosi tekstiksi ja järjestetään sijoituksen mukaan
top5_results['year'] = top5_results['year'].astype(str)
top5_results = top5_results.sort_values(['year', 'place'])

st.header("Tekoälyn ennuste vuoden 2026 sijoituksista")

# Vaihdetaan Classifieriin!
from sklearn.ensemble import RandomForestClassifier

# --- 1. DATAN VALMISTELU ---
ml_data = final_df.dropna(subset=['running_order', 'win_chance', 'place']).copy()

# Muutetaan sijoitus kokonaisluvuksi, jotta luokat ovat selkeitä
ml_data['place'] = ml_data['place'].astype(int)

X = ml_data[['running_order', 'win_chance']]
y = ml_data['place']

# --- 2. MALLIN OPETTAMINEN ---
# Käytetään Classifieria
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.write("Tämä lista on järjestetty sen mukaan, kuinka suuren todennäköisyyden tekoäly antaa maalle menestyä (perustuen kertoimiin ja esiintymispaikkaan).")

# --- 1. EROTETAAN OPETUSDATA JA ENNUSTEDATA ---
# Opetusdata: Kaikki vanhat vuodet, joissa on sijoitus mukana
train_data = final_df.dropna(subset=['place', 'running_order', 'win_chance']).copy()
train_data['place'] = train_data['place'].astype(int)

X_train = train_data[['running_order', 'win_chance']]
y_train = train_data['place']

# Opetetaan malli vanhalla datalla
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ennustedata: Haetaan summary-taulukosta suoraan vuoden 2026 tiedot ennen kuin ne suodatettiin pois!
# Yhdistetään ne contestants-tiedon kanssa, mutta EI tiputeta puuttuvia sijoituksia (dropna) pois
current_year_raw = summary[summary['year'] == 2026].copy()
current_year_data = pd.merge(current_year_raw, results_df[results_df['year'] == 2026], on=['year', 'country_name'])

# Varmistetaan, että running_order on numero
current_year_data['running_order'] = pd.to_numeric(current_year_data['running_order'], errors='coerce')
current_year_data = current_year_data.dropna(subset=['running_order'])

if not current_year_data.empty:
    # --- 2. ENNUSTETAAN VUOSI 2026 ---
    X_current = current_year_data[['running_order', 'win_chance']]
    
    # Lasketaan todennäköisyydet sijoille 1-5
    probabilities = model.predict_proba(X_current)
    top5_classes_indices = [i for i, c in enumerate(model.classes_) if c <= 5]
    current_year_data['success_score'] = probabilities[:, top5_classes_indices].sum(axis=1)
    
    # Järjestetään ennusteen mukaan
    predictions_list = current_year_data.sort_values('success_score', ascending=False).reset_index(drop=True)
    
    # Näytetään tulokset
    for rank, (idx, row) in enumerate(predictions_list.iterrows(), 1):
        
        # Lasketaan prosentti kauniiksi merkkijonoksi
        ai_prosentti = int(round(row['success_score'] * 100))
        
        medal = ""
        if rank == 1: medal = "🥇 "
        elif rank == 2: medal = "🥈 "
        elif rank == 3: medal = "🥉 "
        
        st.write(f"**{medal}{rank}. {row['country_name']}** (Esiintyy paikalla {int(row['running_order'])})")
        # Näytetään vedonlyönnin ja tekoälyn antamat TOP 5 -prosentit rinnakkain!
        st.caption(f"Vedonlyöntitodennäköisyys: {row['win_chance']}% | AI:n arvioima TOP 5 -todennäköisyys: {ai_prosentti}%")
else:
    st.info("Vuoden 2026 dataa ei löytynyt tai siitä puuttuu esiintymisjärjestys.")

# Luodaan selite sijoituksesta
top5_results['Sijoitus'] = top5_results['place'].apply(lambda x: 'Voittaja' if x == 1 else f'{int(x)}. Sija')
# Graafia luodessa käytetään tätä uutta saraketta värinä
fig = px.bar(top5_results, 
             x='year', 
             y='win_chance', 
             color='Sijoitus', # Nyt tässä on eri arvot jokaiselle!
             text='country_name',
             barmode='group',
             title='Vuosittaiset TOP 5 vierekkäin',
             labels={'win_chance': 'Voittomahdollisuus (%)', 'year': 'Vuosi', 'country_name': 'Maa'})

# Jotta graafi ei ole pelkkää sateenkaarta, pakotetaan vuodet kategorioiksi
fig.update_layout(
    height=800, # Nostetaan korkeus esim. 800 tai 1000 pikseliin
    xaxis={'type': 'category'},
    xaxis_type='category', 
    showlegend=False)
fig.update_traces(textposition='outside')

st.plotly_chart(fig, width='stretch')

st.write("---")
st.header("Vuosittaiset TOP 5 ennakkosuosikit (Vedonlyönti)")

# Käytetään tuota uutta top5_favorites -muuttujaa
fav_data = top5_favorites.sort_values(['year', 'win_chance'], ascending=[False, False])

for year in fav_data['year'].unique():
    # Muutetaan vuosi nätiksi otsikoksi
    with st.expander(f"Ennakkosuosikit {int(year)}"):
        year_favs = fav_data[fav_data['year'] == year]
        
        for i, (idx, row) in enumerate(year_favs.iterrows(), 1):
            # Katsotaan miten kävi
            status = "🏆 VOITTAJA" if row['place'] == 1 else f"Lopullinen sija: #{int(row['place'])}"
            
            st.write(f"**{i}. {row['country_name']}** ({row['win_chance']} %)")
            st.caption(f"Artisti: {row['performer']} | {status}")

st.write("---")


    # --- 3. ENNUSTAMINEN STREAMLITISSÄ ---
with st.expander("Testaa mallia: Syötä maan lähtötiedot ja katso tekoälyn ennuste sijoituksesta!"):

    input_order = st.slider("Esiintymispaikka finaalissa (Running Order)", min_value=1, max_value=26, value=1)
    input_chance = st.slider("Vedonlyönnin voittomahdollisuus (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)

    # Ennustetaan suoraan luokka (eli tarkka sijoitus)
    predicted_place = model.predict([[input_order, input_chance]])[0]

    # Näytetään tulos (ei tarvitse enää pyöristellä!)
    st.subheader(f"Ennustettu lopullinen sijoitus: **{predicted_place}. sija**")

    if predicted_place == 1:
        st.balloons() # Laitetaan streamlit-ilmapallot lentämään voiton kunniaksi! 🎉
        st.success("🏆 TEKOÄLY ENNUSTAA VOITTOA! TORILLE! 🇫🇮")
    elif predicted_place <= 5:
        st.success("🔥 Tekoälyn mukaan tämä maa taistelee kärkisijoista!")
    elif predicted_place <= 15:
        st.info("🎵 Keskikastin suoritus tulossa.")
    else:
        st.warning("📉 Tällä esiintymispaikalla ja suosiolla voi tehdä tiukkaa.")