import pandas as pd
import plotly.express as px
import streamlit as st
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
cols = ['year', 'country_code', 'country_name', 'song', 'performer', 'place']
results_df = pd.read_csv('contestants.csv', header=None, usecols=range(6), names=cols)
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

# Luodaan selite sijoituksesta
top5_results['Sijoitus'] = top5_results['place'].apply(lambda x: 'Voittaja' if x == 1 else f'{int(x)}. Sija')
# Graafia luodessa käytetään tätä uutta saraketta värinä
fig = px.bar(top5_results, 
             x='year', 
             y='win_chance', 
             color='Sijoitus', # Nyt tässä on eri arvot jokaiselle!
             text='country_name',
             barmode='group',
             title='Vuosittaiset TOP 5 ennakkosuosikit vierekkäin',
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